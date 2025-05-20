$(cat release/pack.py | awk '{
    if (NR == 35) {
        print "# Define a whitelist of allowed modules"
        print "ALLOWED_MODULES = {"
        print "    # Add all legitimate modules that can be imported"
        print "    # Example: \"selfdrive.car.honda.interface\": \"selfdrive.car.honda.interface\","
        print "    # Customize this list based on your actual requirements"
        print "}"
        print ""
        print "# Check if module_name is in the whitelist before importing"
        print "if module_name in ALLOWED_MODULES:"
        print "    module = importlib.import_module(module_name)"
        print "else:"
        print "    raise ImportError(f\"Importing module \'{module_name}\' is not allowed for security reasons\")"
    } else {
        print $0
    }
}')
