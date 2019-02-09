#!/data/data/com.termux/files/usr/bin/env python2

import os
def files_to_delete(rootfolder):
    return sorted(
        (os.path.join(dirname, filename)
         for dirname, dirnames, filenames in os.walk(rootfolder)
         for filename in filenames),
            key=lambda fn: os.stat(fn).st_mtime),reversed==True
def free_space_up_to(free_bytes_required, rootfolder):
    file_list=files_to_delete(rootfolder)
    #print file_list
    while file_list:
        statv=os.statvfs(rootfolder)
        if statv.f_bfree*statv.f_bsize >= free_bytes_required:
            print statv.f_bfree*statv.f_bsize
            break
        #os.remove(file_list.pop())
        #print file_list.pop()[0]
        if len(file_list[0]) > 0:
            print file_list[0].pop()
        else:
            break

def free_pct_up_to(free_pct_required, rootfolder):
    file_list=files_to_delete(rootfolder)
    #print file_list
    while file_list:
        statv=os.statvfs(rootfolder)
        if int(float(statv.f_bfree)/float(statv.f_blocks) * 100) >= free_pct_required:
            print int(float(statv.f_bfree)/float(statv.f_blocks) * 100)
            #print statv.f_bsize
            #print statv.f_bfree
            #print statv.f_blocks
            break
        #print file_list.pop()[0]
        if len(file_list[0]) > 0:
            fname = file_list[0].pop()
            print fname
            os.remove(fname)
        else:
            break

#free_space_up_to(107374182400, "/data/media/0/realdata/")
free_pct_up_to(40, "/data/media/0/realdata/")
