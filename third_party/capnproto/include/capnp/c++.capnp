# Copyright (c) 2013-2014 Sandstorm Development Group, Inc. and contributors
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

@0xbdf87d7bb8304e81;
$namespace("capnp::annotations");

annotation namespace(file): Text;
annotation name(field, enumerant, struct, enum, interface, method, param, group, union): Text;

annotation allowCancellation(interface, method, file) :Void;
# Indicates that the server-side implementation of a method is allowed to be canceled when the
# client requests cancellation. Without this annotation, once a method call has been delivered to
# the server-side application code, any requests by the client to cancel it will be ignored, and
# the method will run to completion anyway. This applies even for local in-process calls.
#
# This behavior applies specifically to implementations that inherit from the C++ `Foo::Server`
# interface. The annotation won't affect DynamicCapability::Server implementations; they must set
# the cancellation mode at runtime.
#
# When applied to an interface rather than an individual method, the annotation applies to all
# methods in the interface. When applied to a file, it applies to all methods defined in the file.
#
# It's generally recommended that this annotation be applied to all methods. However, when doing
# so, it is important that the server implementation use cancellation-safe code. See:
#
#     https://github.com/capnproto/capnproto/blob/master/kjdoc/tour.md#cancellation
#
# If your code is not cancellation-safe, then allowing cancellation might give a malicious client
# an easy way to induce use-after-free or other bugs in your server, by requesting cancellation
# when not expected.
