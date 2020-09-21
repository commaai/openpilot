from tools.lib.url_file import URLFile
#  This test is used more for local benchmarking, so it probably shouldn't run as CI or something like that


FILE = "https://commadataci.blob.core.windows.net/openpilotci/0375fdf7b1ce594d/2019-06-13--08-32-25/3/fcamera.hevc"
#  FILE = "https://commadataci.blob.core.windows.net/openpilotci/0375fdf7b1ce594d/2019-06-13--08-32-25/3/qlog.bz2"


def test_full_download():
    file_large_download = URLFile(FILE, cache=False)
    response = file_large_download.read()
    print(hash(response))
    print(len(response))


def test_full_download_with_cache():
    file_large_download = URLFile(FILE, cache=True)
    response = file_large_download.read()
    print(hash(response))
    print(len(response))


if __name__ == "__main__":
    import timeit
    print(timeit.timeit("test_full_download()", setup="from __main__ import test_full_download", number=1))
    print(timeit.timeit("test_full_download_with_cache()", setup="from __main__ import test_full_download_with_cache", number=1))
    print(timeit.timeit("test_full_download()", setup="from __main__ import test_full_download", number=1))
    print(timeit.timeit("test_full_download_with_cache()", setup="from __main__ import test_full_download_with_cache", number=1))