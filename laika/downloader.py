import certifi
import ftplib
import hatanaka
import os
import urllib.request
import urllib.error
import pycurl
import re
import time
import tempfile
import socket

from datetime import datetime, timedelta
from urllib.parse import urlparse
from io import BytesIO

from atomicwrites import atomic_write

from laika.ephemeris import EphemerisType
from .constants import SECS_IN_HR, SECS_IN_DAY, SECS_IN_WEEK
from .gps_time import GPSTime
from .helpers import ConstellationId

dir_path = os.path.dirname(os.path.realpath(__file__))

class DownloadFailed(Exception):
  pass


def retryable(f):
  """
  Decorator to allow us to pass multiple URLs from which to download.
  Automatically retry the request with the next URL on failure
  """
  def wrapped(url_bases, *args, **kwargs):
    if isinstance(url_bases, str):
      # only one url passed, don't do the retry thing
      return f(url_bases, *args, **kwargs)

    # not a string, must be a list of url_bases
    for url_base in url_bases:
      try:
        return f(url_base, *args, **kwargs)
      except DownloadFailed as e:
        print(e)
    # none of them succeeded
    raise DownloadFailed("Multiple URL failures attempting to pull file(s)")
  return wrapped


def ftp_connect(url):
  parsed = urlparse(url)
  assert parsed.scheme == 'ftp'
  try:
    domain = parsed.netloc
    ftp = ftplib.FTP(domain, timeout=10)
    ftp.login()
  except (OSError, ftplib.error_perm):
    raise DownloadFailed("Could not connect/auth to: " + domain)
  try:
    ftp.cwd(parsed.path)
  except ftplib.error_perm:
    raise DownloadFailed("Permission failure with folder: " + url)
  return ftp


@retryable
def list_dir(url):
  parsed = urlparse(url)
  if parsed.scheme == 'ftp':
    try:
      ftp = ftp_connect(url)
      return ftp.nlst()
    except ftplib.error_perm:
      raise DownloadFailed("Permission failure listing folder: " + url)
  else:
    # just connect and do simple url parsing
    listing = https_download_file(url)
    urls = re.findall(b"<a href=\"([^\"]+)\">", listing)
    # decode the urls to normal strings. If they are complicated paths, ignore them
    return [name.decode("latin1") for name in urls if name and b"/" not in name[1:]]


def ftp_download_files(url_base, folder_path, cacheDir, filenames):
  """
  Like download file, but more of them. Keeps a persistent FTP connection open
  to be more efficient.
  """
  folder_path_abs = os.path.join(cacheDir, folder_path)

  ftp = ftp_connect(url_base + folder_path)

  filepaths = []
  for filename in filenames:
    # os.path.join will be dumb if filename has a leading /
    # if there is a / in the filename, then it's using a different folder
    filename = filename.lstrip("/")
    if "/" in filename:
      continue
    filepath = os.path.join(folder_path_abs, filename)
    print("pulling from", url_base, "to", filepath)

    if not os.path.isfile(filepath):
      os.makedirs(folder_path_abs, exist_ok=True)
      try:
        ftp.retrbinary('RETR ' + filename, open(filepath, 'wb').write)
      except (ftplib.error_perm):
        raise DownloadFailed("Could not download file from: " + url_base + folder_path + filename)
      except (socket.timeout):
        raise DownloadFailed("Read timed out from: " + url_base + folder_path + filename)
      filepaths.append(filepath)
    else:
      filepaths.append(filepath)
  return filepaths


def http_download_files(url_base, folder_path, cacheDir, filenames):
  """
  Similar to ftp_download_files, attempt to download multiple files faster than
  just downloading them one-by-one.
  Returns a list of filepaths instead of the raw data
  """
  folder_path_abs = os.path.join(cacheDir, folder_path)

  def write_function(disk_path, handle):
    def do_write(data):
      open(disk_path, "wb").write(data)

    return do_write

  fetcher = pycurl.CurlMulti()
  fetcher.setopt(pycurl.M_PIPELINING, 3)
  fetcher.setopt(pycurl.M_MAX_HOST_CONNECTIONS, 64)
  fetcher.setopt(pycurl.M_MAX_TOTAL_CONNECTIONS, 64)
  filepaths = []
  for filename in filenames:
    # os.path.join will be dumb if filename has a leading /
    # if there is a / in the filename, then it's using a different folder
    filename = filename.lstrip("/")
    if "/" in filename:
      continue
    filepath = os.path.join(folder_path_abs, filename)
    if not os.path.isfile(filepath):
      print("pulling from", url_base, "to", filepath)
      os.makedirs(folder_path_abs, exist_ok=True)
      url_path = url_base + folder_path + filename
      handle = pycurl.Curl()
      handle.setopt(pycurl.URL, url_path)
      handle.setopt(pycurl.CONNECTTIMEOUT, 10)
      handle.setopt(pycurl.WRITEFUNCTION, write_function(filepath, handle))
      fetcher.add_handle(handle)
      filepaths.append(filepath)

  requests_processing = len(filepaths)
  timeout = 10.0  # after 10 seconds of nothing happening, restart
  deadline = time.time() + timeout
  while requests_processing and time.time() < deadline:
    while True:
      ret, cur_requests_processing = fetcher.perform()
      if ret != pycurl.E_CALL_MULTI_PERFORM:
        _, success, failed = fetcher.info_read()
        break
    if requests_processing > cur_requests_processing:
      deadline = time.time() + timeout
      requests_processing = cur_requests_processing

    if fetcher.select(1) < 0:
      continue

  # if there are downloads left to be done, repeat, and don't overwrite
  _, requests_processing = fetcher.perform()
  if requests_processing > 0:
    print("some requests stalled, retrying them")
    return http_download_files(url_base, folder_path, cacheDir, filenames)

  return filepaths


def https_download_file(url):
  if 'nasa.gov/' not in url:
    netrc_path = None
    f = None
  elif os.path.isfile(dir_path + '/.netrc'):
    netrc_path = dir_path + '/.netrc'
    f = None
  else:
    try:
      username = os.environ['NASA_USERNAME']
      password = os.environ['NASA_PASSWORD']
      f = tempfile.NamedTemporaryFile()
      netrc = f"machine urs.earthdata.nasa.gov login {username} password {password}"
      f.write(netrc.encode())
      f.flush()
      netrc_path = f.name
    except KeyError:
      raise DownloadFailed('Could not find .netrc file and no NASA_USERNAME and NASA_PASSWORD in environment for urs.earthdata.nasa.gov authentication')

  crl = pycurl.Curl()
  crl.setopt(crl.CAINFO, certifi.where())
  crl.setopt(crl.URL, url)
  crl.setopt(crl.FOLLOWLOCATION, True)
  crl.setopt(crl.SSL_CIPHER_LIST, 'DEFAULT@SECLEVEL=1')
  crl.setopt(crl.COOKIEJAR, '/tmp/cddis_cookies')
  crl.setopt(pycurl.CONNECTTIMEOUT, 10)
  if netrc_path is not None:
    crl.setopt(crl.NETRC_FILE, netrc_path)
    crl.setopt(crl.NETRC, 2)

  buf = BytesIO()
  crl.setopt(crl.WRITEDATA, buf)
  crl.perform()
  response = crl.getinfo(pycurl.RESPONSE_CODE)
  crl.close()
  if f is not None:
    f.close()

  if response != 200:
    raise DownloadFailed('HTTPS error ' + str(response))
  return buf.getvalue()


def ftp_download_file(url):
  try:
    urlf = urllib.request.urlopen(url, timeout=10)
    data_zipped = urlf.read()
    urlf.close()
    return data_zipped
  except urllib.error.URLError as e:
    raise DownloadFailed(e)


@retryable
def download_files(url_base, folder_path, cacheDir, filenames):
  parsed = urlparse(url_base)
  if parsed.scheme == 'ftp':
    return ftp_download_files(url_base, folder_path, cacheDir, filenames)
  else:
    return http_download_files(url_base, folder_path, cacheDir, filenames)


@retryable
def download_file(url_base, folder_path, filename_zipped):
  url = url_base + folder_path + filename_zipped
  print('Downloading ' + url)
  if url.startswith('https'):
    return https_download_file(url)
  if url.startswith('ftp'):
    return ftp_download_file(url)
  raise NotImplementedError('Did find ftp or https preamble')


def download_and_cache_file_return_first_success(url_bases, folder_and_file_names, cache_dir, compression='', overwrite=False, raise_error=False):
  last_error = None
  for folder_path, filename in folder_and_file_names:
    try:
      file = download_and_cache_file(url_bases, folder_path, cache_dir, filename, compression, overwrite)
      return file
    except DownloadFailed as e:
      last_error = e

  if last_error and raise_error:
    raise last_error


def download_and_cache_file(url_base, folder_path: str, cache_dir: str, filename: str, compression='', overwrite=False):
  filename_zipped = filename + compression
  folder_path_abs = os.path.join(cache_dir, folder_path)
  filepath = str(hatanaka.get_decompressed_path(os.path.join(folder_path_abs, filename)))

  filepath_attempt = filepath + '.attempt_time'

  if os.path.exists(filepath_attempt):
    with open(filepath_attempt, 'r') as rf:
      last_attempt_time = float(rf.read())
    if time.time() - last_attempt_time < SECS_IN_HR:
      raise DownloadFailed(f"Too soon to try downloading {folder_path + filename_zipped} from {url_base} again since last attempt")
  if not os.path.isfile(filepath) or overwrite:
    try:
      data_zipped = download_file(url_base, folder_path, filename_zipped)
    except (DownloadFailed, pycurl.error, socket.timeout):
      unix_time = time.time()
      os.makedirs(folder_path_abs, exist_ok=True)
      with atomic_write(filepath_attempt, mode='w', overwrite=True) as wf:
        wf.write(str(unix_time))
      raise DownloadFailed(f"Could not download {folder_path + filename_zipped} from {url_base} ")

    os.makedirs(folder_path_abs, exist_ok=True)
    ephem_bytes = hatanaka.decompress(data_zipped)
    try:
      with atomic_write(filepath, mode='wb', overwrite=overwrite) as f:
        f.write(ephem_bytes)
    except FileExistsError:
      # Only happens when same file is downloaded in parallel by another process.
      pass
  return filepath


# Currently, only GPS and Glonass are supported for daily and hourly data.
CONSTELLATION_NASA_CHAR = {ConstellationId.GPS: 'n', ConstellationId.GLONASS: 'g'}


def download_nav(time: GPSTime, cache_dir, constellation: ConstellationId):
  t = time.as_datetime()
  try:
    if constellation not in CONSTELLATION_NASA_CHAR:
      return None
    c = CONSTELLATION_NASA_CHAR[constellation]
    if GPSTime.from_datetime(datetime.utcnow()) - time > SECS_IN_DAY:
      url_bases = (
        'https://github.com/commaai/gnss-data/raw/master/gnss/data/daily/',
        'https://cddis.nasa.gov/archive/gnss/data/daily/',
      )
      filename = t.strftime(f"brdc%j0.%y{c}")
      folder_path = t.strftime(f'%Y/%j/%y{c}/')
      compression = '.gz' if folder_path >= '2020/335/' else '.Z'
      return download_and_cache_file(url_bases, folder_path, cache_dir+'daily_nav/', filename, compression)
    elif constellation == ConstellationId.GPS:
      url_base = 'https://cddis.nasa.gov/archive/gnss/data/hourly/'
      filename = t.strftime(f"hour%j0.%y{c}")
      folder_path = t.strftime('%Y/%j/')
      compression = '.gz' if folder_path >= '2020/336/' else '.Z'
      return download_and_cache_file(url_base, folder_path, cache_dir+'hourly_nav/', filename, compression)
  except DownloadFailed:
    pass


def download_orbits_gps(time, cache_dir, ephem_types):
  url_bases = (
    'https://github.com/commaai/gnss-data/raw/master/gnss/products/',
    'https://cddis.nasa.gov/archive/gnss/products/',
    'ftp://igs.ign.fr/pub/igs/products/',
  )
  folder_path = "%i/" % time.week
  filenames = []
  time_str = "%i%i" % (time.week, time.day)
  # Download filenames in order of quality. Final -> Rapid -> Ultra-Rapid(newest first)
  if EphemerisType.FINAL_ORBIT in ephem_types and GPSTime.from_datetime(datetime.utcnow()) - time > 3 * SECS_IN_WEEK:
    filenames.append(f"igs{time_str}.sp3")
  if EphemerisType.RAPID_ORBIT in ephem_types:
    filenames.append(f"igr{time_str}.sp3")
  if EphemerisType.ULTRA_RAPID_ORBIT in ephem_types:
    filenames.extend([f"igu{time_str}_18.sp3",
                      f"igu{time_str}_12.sp3",
                      f"igu{time_str}_06.sp3",
                      f"igu{time_str}_00.sp3"])
  folder_file_names = [(folder_path, filename) for filename in filenames]
  return download_and_cache_file_return_first_success(url_bases, folder_file_names, cache_dir+'cddis_products/', compression='.Z')


def download_prediction_orbits_russia_src(gps_time, cache_dir):
  # Download single file that contains Ultra_Rapid predictions for GPS, GLONASS and other constellations
  t = gps_time.as_datetime()
  # Files exist starting at 29-01-2022
  if t < datetime(2022, 1, 29):
    return None
  url_bases = 'https://github.com/commaai/gnss-data-alt/raw/master/MCC/PRODUCTS/'
  folder_path = t.strftime('%y%j/ultra/')
  file_prefix = "Stark_1D_" + t.strftime('%y%m%d')

  # Predictions are 24H so previous day can also be used.
  prev_day = (t - timedelta(days=1))
  file_prefix_prev = "Stark_1D_" + prev_day.strftime('%y%m%d')
  folder_path_prev = prev_day.strftime('%y%j/ultra/')

  current_day = GPSTime.from_datetime(datetime(t.year, t.month, t.day))
  # Ultra-Orbit is published in gnss-data-alt every 10th minute past the 5,11,17,23 hour.
  # Predictions published are delayed by around 10 hours.
  # Download latest file that includes gps_time with 20 minutes margin.:
  if gps_time > current_day + 23.5 * SECS_IN_HR:
    prev_day, current_day = [], [6, 12]
  elif gps_time > current_day + 17.5 * SECS_IN_HR:
    prev_day, current_day = [], [0, 6]
  elif gps_time > current_day + 11.5 * SECS_IN_HR:
    prev_day, current_day = [18], [0]
  elif gps_time > current_day + 5.5 * SECS_IN_HR:
    prev_day, current_day = [12, 18], []
  else:
    prev_day, current_day = [6, 12], []
  # Example: Stark_1D_22060100.sp3
  folder_and_file_names = [(folder_path, file_prefix + f"{h:02}.sp3") for h in reversed(current_day)] + \
                          [(folder_path_prev, file_prefix_prev + f"{h:02}.sp3") for h in reversed(prev_day)]
  return download_and_cache_file_return_first_success(url_bases, folder_and_file_names, cache_dir+'russian_products/', raise_error=True)


def download_orbits_russia_src(time, cache_dir, ephem_types):
  # Orbits from russian source. Contains GPS, GLONASS, GALILEO, BEIDOU
  url_bases = (
    'https://github.com/commaai/gnss-data-alt/raw/master/MCC/PRODUCTS/',
    'ftp://ftp.glonass-iac.ru/MCC/PRODUCTS/',
  )
  t = time.as_datetime()
  folder_paths = []
  current_gps_time = GPSTime.from_datetime(datetime.utcnow())
  filename = "Sta%i%i.sp3" % (time.week, time.day)
  if EphemerisType.FINAL_ORBIT in ephem_types and current_gps_time - time > 2 * SECS_IN_WEEK:
    folder_paths.append(t.strftime('%y%j/final/'))
  if EphemerisType.RAPID_ORBIT in ephem_types:
    folder_paths.append(t.strftime('%y%j/rapid/'))
  if EphemerisType.ULTRA_RAPID_ORBIT in ephem_types:
    folder_paths.append(t.strftime('%y%j/ultra/'))
  folder_file_names = [(folder_path, filename) for folder_path in folder_paths]
  return download_and_cache_file_return_first_success(url_bases, folder_file_names, cache_dir+'russian_products/')


def download_ionex(time, cache_dir):
  t = time.as_datetime()
  url_bases = (
    'https://github.com/commaai/gnss-data/raw/master/gnss/products/ionex/',
    'https://cddis.nasa.gov/archive/gnss/products/ionex/',
    'ftp://igs.ensg.ign.fr/pub/igs/products/ionosphere/',
    'ftp://gssc.esa.int/gnss/products/ionex/',
  )
  folder_path = t.strftime('%Y/%j/')
  filenames = [t.strftime("codg%j0.%yi"), t.strftime("c1pg%j0.%yi"), t.strftime("c2pg%j0.%yi")]

  folder_file_names = [(folder_path, f) for f in filenames]
  return download_and_cache_file_return_first_success(url_bases, folder_file_names, cache_dir+'ionex/', compression='.Z', raise_error=True)


def download_dcb(time, cache_dir):
  filenames = []
  folder_paths = []
  url_bases = (
    'https://github.com/commaai/gnss-data/raw/master/gnss/products/bias/',
    'https://cddis.nasa.gov/archive/gnss/products/bias/',
    'ftp://igs.ign.fr/pub/igs/products/mgex/dcb/',
  )
  # seem to be a lot of data missing, so try many days
  for time_step in [time - i * SECS_IN_DAY for i in range(14)]:
    t = time_step.as_datetime()
    folder_paths.append(t.strftime('%Y/'))
    filenames.append(t.strftime("CAS0MGXRAP_%Y%j0000_01D_01D_DCB.BSX"))

  return download_and_cache_file_return_first_success(url_bases, list(zip(folder_paths, filenames)), cache_dir+'dcb/', compression='.gz', raise_error=True)


def download_cors_coords(cache_dir):
  cache_subdir = cache_dir + 'cors_coord/'
  url_bases = (
    'https://geodesy.noaa.gov/corsdata/coord/coord_14/',
    'https://alt.ngs.noaa.gov/corsdata/coord/coord_14/',
  )
  file_names = list_dir(url_bases)
  file_names = [file_name for file_name in file_names if file_name.endswith('coord.txt')]
  filepaths = download_files(url_bases, '', cache_subdir, file_names)
  return filepaths


def download_cors_station(time, station_name, cache_dir):
  t = time.as_datetime()
  folder_path = t.strftime('%Y/%j/') + station_name + '/'
  filename = station_name + t.strftime("%j0.%yd")
  url_bases = (
    'https://geodesy.noaa.gov/corsdata/rinex/',
    'https://alt.ngs.noaa.gov/corsdata/rinex/',
  )
  try:
    filepath = download_and_cache_file(url_bases, folder_path, cache_dir+'cors_obs/', filename, compression='.gz')
    return filepath
  except DownloadFailed:
    print("File not downloaded, check availability on server.")
    return None
