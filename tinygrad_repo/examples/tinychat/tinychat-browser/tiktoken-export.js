// Force Webpack to copy the WASM
import 'tiktoken/tiktoken_bg.wasm'; 
import { init, get_encoding, encoding_for_model, Tiktoken } from 'tiktoken/init';
import { load } from 'tiktoken/load';
export { init, get_encoding, encoding_for_model, Tiktoken, load };