/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./theme/**/*.html",
    "./docs/**/*.md",
  ],
  theme: {
    fontFamily: {
      sans: ['Inter', 'sans-serif'],
      mono: ['Roboto Mono', 'monospace', 'Courier New', 'sans-serif']
    },
    extend: {
      fontFamily: {
        'monument-extended': ['Monument Extended', 'sans-serif'],
        'inter': ['Inter', 'sans-serif'],
      }
    }
  },
  plugins: [require("@tailwindcss/typography"), require('daisyui')],
  daisyui: {
    themes: [
      {
        comma: {
          "primary": "#173349",
          "primary-content": "#ccd3d8",
          "secondary": "#178644",
          "secondary-content": "#d4e7d8",
          "accent": "#51FF00",
          "accent-content": "#021600",
          "neutral": "#000",
          "neutral-content": "#ffff",
          "base-100": "#fff",
          "base-200": "#dedede",
          "base-300": "#bebebe",
          "base-content": "#000",
          "info": "#173349",
          "info-content": "#fff",
          "success": "#178644",
          "success-content": "#fff",
          "warning": "#DA6F25",
          "warning-content": "#fff",
          "error": "#C92231",
          "error-content": "#fff",
          "--rounded-box": "0rem", // border radius rounded-box utility class, used in card and other large boxes
          "--rounded-btn": "0rem", // border radius rounded-btn utility class, used in buttons and similar element
          "--rounded-badge": "0.01rem", // border radius rounded-badge utility class, used in badges and similar
          "--animation-btn": "0.25s", // duration of animation when you click on button
          "--animation-input": "0.2s", // duration of animation for inputs like checkbox, toggle, radio, etc
          "--btn-focus-scale": "0.95", // scale transform of button when you focus on it
          "--border-btn": "0px", // border width of buttons
          "--tab-border": "1px", // border width of tabs
          "--tab-radius": "0.rem", // border radius of tabs
        },
        comma_dark: {
          "primary": "#173349",
          "primary-content": "#ccd3d8",
          "secondary": "#178644",
          "secondary-content": "#d4e7d8",
          "accent": "#51FF00",
          "accent-content": "#021600",
          "neutral": "#fff",
          "neutral-content": "#000",
          "base-100": "#000",
          "base-200": "#dedede",
          "base-300": "#bebebe",
          "base-content": "#fff",
          "info": "#173349",
          "info-content": "#000",
          "success": "#178644",
          "success-content": "#000",
          "warning": "#DA6F25",
          "warning-content": "#000",
          "error": "#C92231",
          "error-content": "#000",
          "--rounded-box": "0rem", // border radius rounded-box utility class, used in card and other large boxes
          "--rounded-btn": "0rem", // border radius rounded-btn utility class, used in buttons and similar element
          "--rounded-badge": "0.01rem", // border radius rounded-badge utility class, used in badges and similar
          "--animation-btn": "0.25s", // duration of animation when you click on button
          "--animation-input": "0.2s", // duration of animation for inputs like checkbox, toggle, radio, etc
          "--btn-focus-scale": "0.95", // scale transform of button when you focus on it
          "--border-btn": "0px", // border width of buttons
          "--tab-border": "1px", // border width of tabs
          "--tab-radius": "0.rem", // border radius of tabs
        },
      },
    ],
  }
}
