# Copyright (c) 2016 NetDEF, Inc. and contributors
# Licensed under the MIT License:
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

@0xc0183dd65ffef0f3;

annotation nameinfix @0x85a8d86d736ba637 (file): Text;
# add an infix (middle insert) for output file names
#
# "make" generally has implicit rules for compiling "foo.c" => "foo".  This
# is very annoying with capnp since the rule will be "foo" => "foo.c", leading
# to a loop.  $nameinfix (recommended parameter: "-gen") inserts its parameter
# before the ".c", so the filename becomes "foo-gen.c"
#
# ("foo" is really "foo.capnp", so it's foo.capnp-gen.c)

annotation fieldgetset @0xf72bc690355d66de (file): Void;
# generate getter & setter functions for accessing fields
#
# allows grabbing/putting values without de-/encoding the entire struct.
