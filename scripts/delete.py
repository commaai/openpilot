import git_filter_repo as gfr

def blob_callback(blob, keep_blob_callback):
  to_delete = [

  ]
  size_threshold = 100 * 1024  # 100K

  if blob.size > size_threshold: #and blob.original_path.decode('utf8') :
    print("removing", blob.original_path.decode())
    return

  keep_blob_callback()


if __name__ == '__main__':
  from git_filter_repo import Filter, BlobFilter
  f = Filter(blob_callback=blob_callback)
  f.run()
