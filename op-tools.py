import time, sys, json
from os import system, name

def main_menu():
    print("Please chooce a category:")
    while True:
        print("1. Dynamic follow\n2. GPS\n3. Other")
        c = input("[1-3]: ")
        try:
            int(c)
        except:
            return
        if int(c) - 1 in range(3):
            break
    if int(c) == 1:
        dynamic_follow()

def dynamic_follow():
    clear()
    print("\nPress Ctrl + C at any time to quit!")
    time.sleep(3)
    try:
        while True:
            clear()
            print("\nDynamic follow data:\n")
            try:
                with open("/data/op-tools-data/dynamic-follow.json", "r") as f:
                    data = json.loads(f.read())
                    for i in data:
                        print(i + ": " + str(data[i]))
            except:
                print("\nReading error, waiting a second...")
                time.sleep(1)
            time.sleep(.15)
    except KeyboardInterrupt: 
        print()   
        main_menu()

def clear(): 
    # for windows 
    if name == 'nt': 
        system('cls') 
  
    # for mac and linux(here, os.name is 'posix') 
    else: 
        system('clear') 
    

main_menu()


'''while True:
    sys.stdout.write("\r\r\r" + time.ctime()+"\n"+time.ctime())
    sys.stdout.flush()
    time.sleep(1)'''