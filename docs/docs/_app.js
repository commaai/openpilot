export const config = {
  unstable_runtimeJS: false
};

// This default export is required in a new `pages/_app.js` file.
function MyApp({ Component, pageProps }) {
  return <Component {...pageProps} />
}

export default MyApp;
