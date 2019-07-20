from selfdrive.kegman_conf import kegman_conf
import subprocess
import os
from common.params import Params

BASEDIR = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))

letters = { "a":[ "###", "# #", "###", "# #", "# #"], "b":[ "###", "# #", "###", "# #", "###"], "c":[ "###", "#", "#", "#", "###"], "d":[ "##", "# #", "# #", "# #", "##"], "e":[ "###", "#", "###", "#", "###"], "f":[ "###", "#", "###", "#", "#"], "g":[ "###", "# #", "###", "  #", "###"], "h":[ "# #", "# #", "###", "# #", "# #"], "i":[ "###", " #", " #", " #", "###"], "j":[ "###", " #", " #", " #", "##"], "k":[ "# #", "##", "#", "##", "# #"], "l":[ "#", "#", "#", "#", "###"], "m":[ "# #", "###", "###", "# #", "# #"], "n":[ "###", "# #", "# #", "# #", "# #"], "o":[ "###", "# #", "# #", "# #", "###"], "p":[ "###", "# #", "###", "#", "#"], "q":[ "###", "# #", "###", "  #", "  #"], "r":[ "###", "# #", "##", "# #", "# #"], "s":[ "###", "#", "###", "  #", "###"], "t":[ "###", " #", " #", " #", " #"], "u":[ "# #", "# #", "# #", "# #", "###"], "v":[ "# #", "# #", "# #", "# #", " #"], "w":[ "# #", "# #", "# #", "###", "###"], "x":[ "# #", " #", " #", " #", "# #"], "y":[ "# #", "# #", "###", "  #", "###"], "z":[ "###", "  #", " #", "#", "###"], " ":[ " "], "1":[ " #", "##", " #", " #", "###"], "2":[ "###", "  #", "###", "#", "###"], "3":[ "###", "  #", "###", "  #", "###"], "4":[ "#", "#", "# #", "###", "  #"], "5":[ "###", "#", "###", "  #", "###"], "6":[ "###", "#", "###", "# #", "###"], "7":[ "###", "  # ", " #", " #", "#"], "8":[ "###", "# #", "###", "# #", "###"], "9":[ "###", "# #", "###", "  #", "###"], "0":[ "###", "# #", "# #", "# #", "###"], "!":[ " # ", " # ", " # ", "   ", " # "], "?":[ "###", "  #", " ##", "   ", " # "], ".":[ "   ", "   ", "   ", "   ", " # "], "]":[ "   ", "   ", "   ", "  #", " # "], "/":[ "  #", "  #", " # ", "# ", "# "], ":":[ "   ", " # ", "   ", " # ", "   "], "@":[ "###", "# #", "## ", "#  ", "###"], "'":[ " # ", " # ", "   ", "   ", "   "], "#":[ " # ", "###", " # ", "###", " # "], "-":[ "  ", "  ","###","   ","   "] }
# letters stolen from here: http://www.stuffaboutcode.com/2013/08/raspberry-pi-minecraft-twitter.html

def print_letters(text):
    bigletters = []
    for i in text:
        bigletters.append(letters.get(i.lower(),letters[' ']))
    output = ['']*5
    for i in range(5):
        for j in bigletters:
            temp = ' '
            try:
                temp = j[i]
            except:
                pass
            temp += ' '*(5-len(temp))
            temp = temp.replace(' ',' ')
            temp = temp.replace('#','@')
            output[i] += temp
    return '\n'.join(output)
import sys, termios, tty, os, time

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

button_delay = 0.2

kegman = kegman_conf()
#kegman.conf['tuneGernby'] = "1"
#kegman.write_config(kegman.conf)
param = ["Kp", "Ki", "reactMPC", "dampMPC", "dampTime", "polyFactor", "polyReact", "polyDamp"]

j = 0
while True:
  print ""
  print print_letters(param[j][0:9])
  print ""
  print print_letters(kegman.conf[param[j]])
  print ""
  print ("1, 3, 5, 7, 9 to incr 0.2, 0.01, 0.005, 0.0002, 0.00001")
  print ("a, d, g, j, l (L) to decr 0.2, 0.01, 0.005, 0.0002, 0.00001")
  print ("press SPACE / m for next /prev parameter")
  print ("press z to quit")

  char  = getch()
  write_json = False
  if (char == "9"):
    kegman.conf[param[j]] = str(float(kegman.conf[param[j]]) + 0.00001)
    write_json = True

  if (char == "7"):
    kegman.conf[param[j]] = str(float(kegman.conf[param[j]]) + 0.0002)
    write_json = True

  if (char == "5"):
    kegman.conf[param[j]] = str(float(kegman.conf[param[j]]) + 0.005)
    write_json = True

  elif (char == "3"):
    kegman.conf[param[j]] = str(float(kegman.conf[param[j]]) + 0.01)
    write_json = True

  elif (char == "1"):
    kegman.conf[param[j]] = str(float(kegman.conf[param[j]]) + 0.2)
    write_json = True

  elif (char == "l"):
    kegman.conf[param[j]] = str(float(kegman.conf[param[j]]) - 0.00001)
    write_json = True

  elif (char == "j"):
    kegman.conf[param[j]] = str(float(kegman.conf[param[j]]) - 0.0002)
    write_json = True

  elif (char == "g"):
    kegman.conf[param[j]] = str(float(kegman.conf[param[j]]) - 0.005)
    write_json = True

  elif (char == "d"):
    kegman.conf[param[j]] = str(float(kegman.conf[param[j]]) - 0.01)
    write_json = True

  elif (char == "a"):
    kegman.conf[param[j]] = str(float(kegman.conf[param[j]]) - 0.2)
    write_json = True

  elif (char == " "):
    if j < len(param) - 1:
      j = j + 1
    else:
      j = 0

  elif (char == "m"):
    if j > 0:
      j = j - 1
    else:
      j = len(param) - 1

  elif (char == "z"):
    process.kill()
    break


  if float(kegman.conf['Ki']) < 0 and float(kegman.conf['Ki']) != -1:
    kegman.conf['Ki'] = "0"

  if float(kegman.conf['Ki']) > 2:
    kegman.conf['Ki'] = "2"

  if float(kegman.conf['Kp']) < 0 and float(kegman.conf['Kp']) != -1:
    kegman.conf['Kp'] = "0"

  if float(kegman.conf['Kp']) > 3:
    kegman.conf['Kp'] = "3"


  if write_json:
    kegman.write_config(kegman.conf)

  time.sleep(button_delay)
else:
  process.kill()
