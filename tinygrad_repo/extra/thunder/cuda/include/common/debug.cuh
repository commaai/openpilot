#pragma once

// Reset
#define TK_RESET "\033[0m"

// Foreground colors
#define TK_FG_BLACK "\033[30m"
#define TK_FG_RED "\033[31m"
#define TK_FG_GREEN "\033[32m"
#define TK_FG_YELLOW "\033[33m"
#define TK_FG_BLUE "\033[34m"
#define TK_FG_MAGENTA "\033[35m"
#define TK_FG_CYAN "\033[36m"
#define TK_FG_WHITE "\033[37m"

// Background colors
#define TK_BG_BLACK "\033[40m"
#define TK_BG_RED "\033[41m"
#define TK_BG_GREEN "\033[42m"
#define TK_BG_YELLOW "\033[43m"
#define TK_BG_BLUE "\033[44m"
#define TK_BG_MAGENTA "\033[45m"
#define TK_BG_CYAN "\033[46m"
#define TK_BG_WHITE "\033[47m"

// Bright foreground colors
#define TK_FG_BRIGHT_BLACK "\033[90m"
#define TK_FG_BRIGHT_RED "\033[91m"
#define TK_FG_BRIGHT_GREEN "\033[92m"
#define TK_FG_BRIGHT_YELLOW "\033[93m"
#define TK_FG_BRIGHT_BLUE "\033[94m"
#define TK_FG_BRIGHT_MAGENTA "\033[95m"
#define TK_FG_BRIGHT_CYAN "\033[96m"
#define TK_FG_BRIGHT_WHITE "\033[97m"

// Bright background colors
#define TK_BG_BRIGHT_BLACK "\033[100m"
#define TK_BG_BRIGHT_RED "\033[101m"
#define TK_BG_BRIGHT_GREEN "\033[102m"
#define TK_BG_BRIGHT_YELLOW "\033[103m"
#define TK_BG_BRIGHT_BLUE "\033[104m"
#define TK_BG_BRIGHT_MAGENTA "\033[105m"
#define TK_BG_BRIGHT_CYAN "\033[106m"
#define TK_BG_BRIGHT_WHITE "\033[107m"

// Text styles
#define TK_BOLD "\033[1m"
#define TK_DIM "\033[2m"
#define TK_ITALIC "\033[3m"
#define TK_UNDERLINE "\033[4m"
#define TK_BLINK "\033[5m"
#define TK_REVERSE "\033[7m"
#define TK_HIDDEN "\033[8m"

// Macro to combine styles
#define TK_STYLE(...)  "\033[" #__VA_ARGS__ "m"