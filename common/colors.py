class COLORS:
  def __init__(self):
    self.HEADER = '\033[95m'
    self.OKBLUE = '\033[94m'
    self.CBLUE = '\33[44m'
    self.BOLD = '\033[1m'
    self.CITALIC = '\33[3m'
    self.OKGREEN = '\033[92m'
    self.CWHITE = '\33[37m'
    self.ENDC = '\033[0m' + self.CWHITE
    self.UNDERLINE = '\033[4m'
    self.PINK = '\33[38;5;207m'
    self.PRETTY_YELLOW = self.BASE(220)

    self.RED = '\033[91m'
    self.PURPLE_BG = '\33[45m'
    self.YELLOW = '\033[93m'
    self.BLUE_GREEN = self.BASE(85)

    self.FAIL = self.RED
    # self.INFO = self.PURPLE_BG
    self.INFO = self.BASE(207)
    self.SUCCESS = self.OKGREEN
    self.PROMPT = self.YELLOW
    self.DBLUE = '\033[36m'
    self.CYAN = self.BASE(39)
    self.WARNING = '\033[33m'

  def BASE(self, col):  # seems to support more colors
    return '\33[38;5;{}m'.format(col)

  def BASEBG(self, col):  # seems to support more colors
    return '\33[48;5;{}m'.format(col)


COLORS = COLORS()
