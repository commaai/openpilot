#[macro_use] extern crate tera;

use std::collections::HashMap;

use tera::{Result};
use serde_json::{Value, to_value};

use std::fs::File;
use std::io::Write;
use std::{io};
use std::io::Read;
use std::process;
use std::env;

pub fn do_nothing_filter(value: Value, _: HashMap<String, Value>) -> Result<Value> {
    let s = try_get_value!("do_nothing_filter", "value", String, value);
    Ok(to_value(&s).unwrap())
}

fn main() -> io::Result<()> {
    // read command line arguments
    let args: Vec<String> = env::args().collect();

    let template_glob = &args[1]; // relative glob to template file
    let template_file = &args[2]; // template file path relative 
                                  // to 'template_glob'

    let json_file     = &args[3]; // relative path json file
    let out_file      = &args[4]; // relative path to output file

    // print arguments
    // println!("template path: {}", template_glob);
    // println!("template file: {}", template_file);
    // println!("json file: {}"    , json_file);
    // println!("out file: {}"     , out_file);

    let mut file = File::open(json_file)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    // println!("{}", contents);
    
    // Parse the string of data into serde_json::Value.
    let v: Value = serde_json::from_str(&contents)?;

    let mut tera = compile_templates!(template_glob);
    tera.autoescape_on(vec![".swp"]);
    tera.register_filter("do_nothing", do_nothing_filter);

    match tera.render(template_file, &v) {
        Ok(s) => {
            let mut f_out = File::create(out_file).expect("Unable to create file");
            f_out.write_all(s.as_bytes())?;
        },
        Err(e) => {
            println!("Error: {}", e);
            for e in e.iter().skip(1) {
                println!("Reason: {}", e);
            }
            process::exit(1);
        }
    };

    // println!("-> successfully rendered template!\n");
    Ok(())
}
