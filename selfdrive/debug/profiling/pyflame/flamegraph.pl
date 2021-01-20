#!/usr/bin/perl -w
#
# flamegraph.pl		flame stack grapher.
#
# This takes stack samples and renders a call graph, allowing hot functions
# and codepaths to be quickly identified.  Stack samples can be generated using
# tools such as DTrace, perf, SystemTap, and Instruments.
#
# USAGE: ./flamegraph.pl [options] input.txt > graph.svg
#
#        grep funcA input.txt | ./flamegraph.pl [options] > graph.svg
#
# Then open the resulting .svg in a web browser, for interactivity: mouse-over
# frames for info, click to zoom, and ctrl-F to search.
#
# Options are listed in the usage message (--help).
#
# The input is stack frames and sample counts formatted as single lines.  Each
# frame in the stack is semicolon separated, with a space and count at the end
# of the line.  These can be generated for Linux perf script output using
# stackcollapse-perf.pl, for DTrace using stackcollapse.pl, and for other tools
# using the other stackcollapse programs.  Example input:
#
#  swapper;start_kernel;rest_init;cpu_idle;default_idle;native_safe_halt 1
#
# An optional extra column of counts can be provided to generate a differential
# flame graph of the counts, colored red for more, and blue for less.  This
# can be useful when using flame graphs for non-regression testing.
# See the header comment in the difffolded.pl program for instructions.
#
# The input functions can optionally have annotations at the end of each
# function name, following a precedent by some tools (Linux perf's _[k]):
# 	_[k] for kernel
#	_[i] for inlined
#	_[j] for jit
#	_[w] for waker
# Some of the stackcollapse programs support adding these annotations, eg,
# stackcollapse-perf.pl --kernel --jit. They are used merely for colors by
# some palettes, eg, flamegraph.pl --color=java.
#
# The output flame graph shows relative presence of functions in stack samples.
# The ordering on the x-axis has no meaning; since the data is samples, time
# order of events is not known.  The order used sorts function names
# alphabetically.
#
# While intended to process stack samples, this can also process stack traces.
# For example, tracing stacks for memory allocation, or resource usage.  You
# can use --title to set the title to reflect the content, and --countname
# to change "samples" to "bytes" etc.
#
# There are a few different palettes, selectable using --color.  By default,
# the colors are selected at random (except for differentials).  Functions
# called "-" will be printed gray, which can be used for stack separators (eg,
# between user and kernel stacks).
#
# HISTORY
#
# This was inspired by Neelakanth Nadgir's excellent function_call_graph.rb
# program, which visualized function entry and return trace events.  As Neel
# wrote: "The output displayed is inspired by Roch's CallStackAnalyzer which
# was in turn inspired by the work on vftrace by Jan Boerhout".  See:
# https://blogs.oracle.com/realneel/entry/visualizing_callstacks_via_dtrace_and
#
# Copyright 2016 Netflix, Inc.
# Copyright 2011 Joyent, Inc.  All rights reserved.
# Copyright 2011 Brendan Gregg.  All rights reserved.
#
# CDDL HEADER START
#
# The contents of this file are subject to the terms of the
# Common Development and Distribution License (the "License").
# You may not use this file except in compliance with the License.
#
# You can obtain a copy of the license at docs/cddl1.txt or
# http://opensource.org/licenses/CDDL-1.0.
# See the License for the specific language governing permissions
# and limitations under the License.
#
# When distributing Covered Code, include this CDDL HEADER in each
# file and include the License file at docs/cddl1.txt.
# If applicable, add the following below this CDDL HEADER, with the
# fields enclosed by brackets "[]" replaced with your own identifying
# information: Portions Copyright [yyyy] [name of copyright owner]
#
# CDDL HEADER END
#
# 11-Oct-2014	Adrien Mahieux	Added zoom.
# 21-Nov-2013   Shawn Sterling  Added consistent palette file option
# 17-Mar-2013   Tim Bunce       Added options and more tunables.
# 15-Dec-2011	Dave Pacheco	Support for frames with whitespace.
# 10-Sep-2011	Brendan Gregg	Created this.

use strict;

use Getopt::Long;

use open qw(:std :utf8);

# tunables
my $encoding;
my $fonttype = "Verdana";
my $imagewidth = 1200;          # max width, pixels
my $frameheight = 16;           # max height is dynamic
my $fontsize = 12;              # base text size
my $fontwidth = 0.59;           # avg width relative to fontsize
my $minwidth = 0.1;             # min function width, pixels
my $nametype = "Function:";     # what are the names in the data?
my $countname = "samples";      # what are the counts in the data?
my $colors = "hot";             # color theme
my $bgcolor1 = "#eeeeee";       # background color gradient start
my $bgcolor2 = "#eeeeb0";       # background color gradient stop
my $nameattrfile;               # file holding function attributes
my $timemax;                    # (override the) sum of the counts
my $factor = 1;                 # factor to scale counts by
my $hash = 0;                   # color by function name
my $palette = 0;                # if we use consistent palettes (default off)
my %palette_map;                # palette map hash
my $pal_file = "palette.map";   # palette map file name
my $stackreverse = 0;           # reverse stack order, switching merge end
my $inverted = 0;               # icicle graph
my $negate = 0;                 # switch differential hues
my $titletext = "";             # centered heading
my $titledefault = "Flame Graph";	# overwritten by --title
my $titleinverted = "Icicle Graph";	#   "    "
my $searchcolor = "rgb(230,0,230)";	# color for search highlighting
my $notestext = "";		# embedded notes in SVG
my $subtitletext = "";		# second level title (optional)
my $help = 0;

sub usage {
	die <<USAGE_END;
USAGE: $0 [options] infile > outfile.svg\n
	--title TEXT     # change title text
	--subtitle TEXT  # second level title (optional)
	--width NUM      # width of image (default 1200)
	--height NUM     # height of each frame (default 16)
	--minwidth NUM   # omit smaller functions (default 0.1 pixels)
	--fonttype FONT  # font type (default "Verdana")
	--fontsize NUM   # font size (default 12)
	--countname TEXT # count type label (default "samples")
	--nametype TEXT  # name type label (default "Function:")
	--colors PALETTE # set color palette. choices are: hot (default), mem,
	                 # io, wakeup, chain, java, js, perl, red, green, blue,
	                 # aqua, yellow, purple, orange
	--hash           # colors are keyed by function name hash
	--cp             # use consistent palette (palette.map)
	--reverse        # generate stack-reversed flame graph
	--inverted       # icicle graph
	--negate         # switch differential hues (blue<->red)
	--notes TEXT     # add notes comment in SVG (for debugging)
	--help           # this message

	eg,
	$0 --title="Flame Graph: malloc()" trace.txt > graph.svg
USAGE_END
}

GetOptions(
	'fonttype=s'  => \$fonttype,
	'width=i'     => \$imagewidth,
	'height=i'    => \$frameheight,
	'encoding=s'  => \$encoding,
	'fontsize=f'  => \$fontsize,
	'fontwidth=f' => \$fontwidth,
	'minwidth=f'  => \$minwidth,
	'title=s'     => \$titletext,
	'subtitle=s'  => \$subtitletext,
	'nametype=s'  => \$nametype,
	'countname=s' => \$countname,
	'nameattr=s'  => \$nameattrfile,
	'total=s'     => \$timemax,
	'factor=f'    => \$factor,
	'colors=s'    => \$colors,
	'hash'        => \$hash,
	'cp'          => \$palette,
	'reverse'     => \$stackreverse,
	'inverted'    => \$inverted,
	'negate'      => \$negate,
	'notes=s'     => \$notestext,
	'help'        => \$help,
) or usage();
$help && usage();

# internals
my $ypad1 = $fontsize * 3;      # pad top, include title
my $ypad2 = $fontsize * 2 + 10; # pad bottom, include labels
my $ypad3 = $fontsize * 2;      # pad top, include subtitle (optional)
my $xpad = 10;                  # pad lefm and right
my $framepad = 1;		# vertical padding for frames
my $depthmax = 0;
my %Events;
my %nameattr;

if ($titletext eq "") {
	unless ($inverted) {
		$titletext = $titledefault;
	} else {
		$titletext = $titleinverted;
	}
}

if ($nameattrfile) {
	# The name-attribute file format is a function name followed by a tab then
	# a sequence of tab separated name=value pairs.
	open my $attrfh, $nameattrfile or die "Can't read $nameattrfile: $!\n";
	while (<$attrfh>) {
		chomp;
		my ($funcname, $attrstr) = split /\t/, $_, 2;
		die "Invalid format in $nameattrfile" unless defined $attrstr;
		$nameattr{$funcname} = { map { split /=/, $_, 2 } split /\t/, $attrstr };
	}
}

if ($notestext =~ /[<>]/) {
	die "Notes string can't contain < or >"
}

# background colors:
# - yellow gradient: default (hot, java, js, perl)
# - blue gradient: mem, chain
# - gray gradient: io, wakeup, flat colors (red, green, blue, ...)
if ($colors eq "mem" or $colors eq "chain") {
	$bgcolor1 = "#eeeeee"; $bgcolor2 = "#e0e0ff";
}
if ($colors =~ /^(io|wakeup|red|green|blue|aqua|yellow|purple|orange)$/) {
	$bgcolor1 = "#f8f8f8"; $bgcolor2 = "#e8e8e8";
}

# SVG functions
{ package SVG;
	sub new {
		my $class = shift;
		my $self = {};
		bless ($self, $class);
		return $self;
	}

	sub header {
		my ($self, $w, $h) = @_;
		my $enc_attr = '';
		if (defined $encoding) {
			$enc_attr = qq{ encoding="$encoding"};
		}
		$self->{svg} .= <<SVG;
<?xml version="1.0"$enc_attr standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg version="1.1" width="$w" height="$h" onload="init(evt)" viewBox="0 0 $w $h" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<!-- Flame graph stack visualization. See https://github.com/brendangregg/FlameGraph for latest version, and http://www.brendangregg.com/flamegraphs.html for examples. -->
<!-- NOTES: $notestext -->
SVG
	}

	sub include {
		my ($self, $content) = @_;
		$self->{svg} .= $content;
	}

	sub colorAllocate {
		my ($self, $r, $g, $b) = @_;
		return "rgb($r,$g,$b)";
	}

	sub group_start {
		my ($self, $attr) = @_;

		my @g_attr = map {
			exists $attr->{$_} ? sprintf(qq/$_="%s"/, $attr->{$_}) : ()
		} qw(class style onmouseover onmouseout onclick);
		push @g_attr, $attr->{g_extra} if $attr->{g_extra};
		$self->{svg} .= sprintf qq/<g %s>\n/, join(' ', @g_attr);

		$self->{svg} .= sprintf qq/<title>%s<\/title>/, $attr->{title}
			if $attr->{title}; # should be first element within g container

		if ($attr->{href}) {
			my @a_attr;
			push @a_attr, sprintf qq/xlink:href="%s"/, $attr->{href} if $attr->{href};
			# default target=_top else links will open within SVG <object>
			push @a_attr, sprintf qq/target="%s"/, $attr->{target} || "_top";
			push @a_attr, $attr->{a_extra}                           if $attr->{a_extra};
			$self->{svg} .= sprintf qq/<a %s>/, join(' ', @a_attr);
		}
	}

	sub group_end {
		my ($self, $attr) = @_;
		$self->{svg} .= qq/<\/a>\n/ if $attr->{href};
		$self->{svg} .= qq/<\/g>\n/;
	}

	sub filledRectangle {
		my ($self, $x1, $y1, $x2, $y2, $fill, $extra) = @_;
		$x1 = sprintf "%0.1f", $x1;
		$x2 = sprintf "%0.1f", $x2;
		my $w = sprintf "%0.1f", $x2 - $x1;
		my $h = sprintf "%0.1f", $y2 - $y1;
		$extra = defined $extra ? $extra : "";
		$self->{svg} .= qq/<rect x="$x1" y="$y1" width="$w" height="$h" fill="$fill" $extra \/>\n/;
	}

	sub stringTTF {
		my ($self, $color, $font, $size, $angle, $x, $y, $str, $loc, $extra) = @_;
		$x = sprintf "%0.2f", $x;
		$loc = defined $loc ? $loc : "left";
		$extra = defined $extra ? $extra : "";
		$self->{svg} .= qq/<text text-anchor="$loc" x="$x" y="$y" font-size="$size" font-family="$font" fill="$color" $extra >$str<\/text>\n/;
	}

	sub svg {
		my $self = shift;
		return "$self->{svg}</svg>\n";
	}
	1;
}

sub namehash {
	# Generate a vector hash for the name string, weighting early over
	# later characters. We want to pick the same colors for function
	# names across different flame graphs.
	my $name = shift;
	my $vector = 0;
	my $weight = 1;
	my $max = 1;
	my $mod = 10;
	# if module name present, trunc to 1st char
	$name =~ s/.(.*?)`//;
	foreach my $c (split //, $name) {
		my $i = (ord $c) % $mod;
		$vector += ($i / ($mod++ - 1)) * $weight;
		$max += 1 * $weight;
		$weight *= 0.70;
		last if $mod > 12;
	}
	return (1 - $vector / $max)
}

sub color {
	my ($type, $hash, $name) = @_;
	my ($v1, $v2, $v3);

	if ($hash) {
		$v1 = namehash($name);
		$v2 = $v3 = namehash(scalar reverse $name);
	} else {
		$v1 = rand(1);
		$v2 = rand(1);
		$v3 = rand(1);
	}

	# theme palettes
	if (defined $type and $type eq "hot") {
		my $r = 205 + int(50 * $v3);
		my $g = 0 + int(230 * $v1);
		my $b = 0 + int(55 * $v2);
		return "rgb($r,$g,$b)";
	}
	if (defined $type and $type eq "mem") {
		my $r = 0;
		my $g = 190 + int(50 * $v2);
		my $b = 0 + int(210 * $v1);
		return "rgb($r,$g,$b)";
	}
	if (defined $type and $type eq "io") {
		my $r = 80 + int(60 * $v1);
		my $g = $r;
		my $b = 190 + int(55 * $v2);
		return "rgb($r,$g,$b)";
	}

	# multi palettes
	if (defined $type and $type eq "java") {
		# Handle both annotations (_[j], _[i], ...; which are
		# accurate), as well as input that lacks any annotations, as
		# best as possible. Without annotations, we get a little hacky
		# and match on java|org|com, etc.
		if ($name =~ m:_\[j\]$:) {	# jit annotation
			$type = "green";
		} elsif ($name =~ m:_\[i\]$:) {	# inline annotation
			$type = "aqua";
		} elsif ($name =~ m:^L?(java|org|com|io|sun)/:) {	# Java
			$type = "green";
		} elsif ($name =~ /::/) {	# C++
			$type = "yellow";
		} elsif ($name =~ m:_\[k\]$:) {	# kernel annotation
			$type = "orange";
		} else {			# system
			$type = "red";
		}
		# fall-through to color palettes
	}
	if (defined $type and $type eq "perl") {
		if ($name =~ /::/) {		# C++
			$type = "yellow";
		} elsif ($name =~ m:Perl: or $name =~ m:\.pl:) {	# Perl
			$type = "green";
		} elsif ($name =~ m:_\[k\]$:) {	# kernel
			$type = "orange";
		} else {			# system
			$type = "red";
		}
		# fall-through to color palettes
	}
	if (defined $type and $type eq "js") {
		# Handle both annotations (_[j], _[i], ...; which are
		# accurate), as well as input that lacks any annotations, as
		# best as possible. Without annotations, we get a little hacky,
		# and match on a "/" with a ".js", etc.
		if ($name =~ m:_\[j\]$:) {	# jit annotation
			if ($name =~ m:/:) {
				$type = "green";	# source
			} else {
				$type = "aqua";		# builtin
			}
		} elsif ($name =~ /::/) {	# C++
			$type = "yellow";
		} elsif ($name =~ m:/.*\.js:) {	# JavaScript (match "/" in path)
			$type = "green";
		} elsif ($name =~ m/:/) {	# JavaScript (match ":" in builtin)
			$type = "aqua";
		} elsif ($name =~ m/^ $/) {	# Missing symbol
			$type = "green";
		} elsif ($name =~ m:_\[k\]:) {	# kernel
			$type = "orange";
		} else {			# system
			$type = "red";
		}
		# fall-through to color palettes
	}
	if (defined $type and $type eq "wakeup") {
		$type = "aqua";
		# fall-through to color palettes
	}
	if (defined $type and $type eq "chain") {
		if ($name =~ m:_\[w\]:) {	# waker
			$type = "aqua"
		} else {			# off-CPU
			$type = "blue";
		}
		# fall-through to color palettes
	}

	# color palettes
	if (defined $type and $type eq "red") {
		my $r = 200 + int(55 * $v1);
		my $x = 50 + int(80 * $v1);
		return "rgb($r,$x,$x)";
	}
	if (defined $type and $type eq "green") {
		my $g = 200 + int(55 * $v1);
		my $x = 50 + int(60 * $v1);
		return "rgb($x,$g,$x)";
	}
	if (defined $type and $type eq "blue") {
		my $b = 205 + int(50 * $v1);
		my $x = 80 + int(60 * $v1);
		return "rgb($x,$x,$b)";
	}
	if (defined $type and $type eq "yellow") {
		my $x = 175 + int(55 * $v1);
		my $b = 50 + int(20 * $v1);
		return "rgb($x,$x,$b)";
	}
	if (defined $type and $type eq "purple") {
		my $x = 190 + int(65 * $v1);
		my $g = 80 + int(60 * $v1);
		return "rgb($x,$g,$x)";
	}
	if (defined $type and $type eq "aqua") {
		my $r = 50 + int(60 * $v1);
		my $g = 165 + int(55 * $v1);
		my $b = 165 + int(55 * $v1);
		return "rgb($r,$g,$b)";
	}
	if (defined $type and $type eq "orange") {
		my $r = 190 + int(65 * $v1);
		my $g = 90 + int(65 * $v1);
		return "rgb($r,$g,0)";
	}

	return "rgb(0,0,0)";
}

sub color_scale {
	my ($value, $max) = @_;
	my ($r, $g, $b) = (255, 255, 255);
	$value = -$value if $negate;
	if ($value > 0) {
		$g = $b = int(210 * ($max - $value) / $max);
	} elsif ($value < 0) {
		$r = $g = int(210 * ($max + $value) / $max);
	}
	return "rgb($r,$g,$b)";
}

sub color_map {
	my ($colors, $func) = @_;
	if (exists $palette_map{$func}) {
		return $palette_map{$func};
	} else {
		$palette_map{$func} = color($colors, $hash, $func);
		return $palette_map{$func};
	}
}

sub write_palette {
	open(FILE, ">$pal_file");
	foreach my $key (sort keys %palette_map) {
		print FILE $key."->".$palette_map{$key}."\n";
	}
	close(FILE);
}

sub read_palette {
	if (-e $pal_file) {
	open(FILE, $pal_file) or die "can't open file $pal_file: $!";
	while ( my $line = <FILE>) {
		chomp($line);
		(my $key, my $value) = split("->",$line);
		$palette_map{$key}=$value;
	}
	close(FILE)
	}
}

my %Node;	# Hash of merged frame data
my %Tmp;

# flow() merges two stacks, storing the merged frames and value data in %Node.
sub flow {
	my ($last, $this, $v, $d) = @_;

	my $len_a = @$last - 1;
	my $len_b = @$this - 1;

	my $i = 0;
	my $len_same;
	for (; $i <= $len_a; $i++) {
		last if $i > $len_b;
		last if $last->[$i] ne $this->[$i];
	}
	$len_same = $i;

	for ($i = $len_a; $i >= $len_same; $i--) {
		my $k = "$last->[$i];$i";
		# a unique ID is constructed from "func;depth;etime";
		# func-depth isn't unique, it may be repeated later.
		$Node{"$k;$v"}->{stime} = delete $Tmp{$k}->{stime};
		if (defined $Tmp{$k}->{delta}) {
			$Node{"$k;$v"}->{delta} = delete $Tmp{$k}->{delta};
		}
		delete $Tmp{$k};
	}

	for ($i = $len_same; $i <= $len_b; $i++) {
		my $k = "$this->[$i];$i";
		$Tmp{$k}->{stime} = $v;
		if (defined $d) {
			$Tmp{$k}->{delta} += $i == $len_b ? $d : 0;
		}
	}

        return $this;
}

# parse input
my @Data;
my $last = [];
my $time = 0;
my $delta = undef;
my $ignored = 0;
my $line;
my $maxdelta = 1;

# reverse if needed
foreach (<>) {
	chomp;
	$line = $_;
	if ($stackreverse) {
		# there may be an extra samples column for differentials
		# XXX todo: redo these REs as one. It's repeated below.
		my($stack, $samples) = (/^(.*)\s+?(\d+(?:\.\d*)?)$/);
		my $samples2 = undef;
		if ($stack =~ /^(.*)\s+?(\d+(?:\.\d*)?)$/) {
			$samples2 = $samples;
			($stack, $samples) = $stack =~ (/^(.*)\s+?(\d+(?:\.\d*)?)$/);
			unshift @Data, join(";", reverse split(";", $stack)) . " $samples $samples2";
		} else {
			unshift @Data, join(";", reverse split(";", $stack)) . " $samples";
		}
	} else {
		unshift @Data, $line;
	}
}

# process and merge frames
foreach (sort @Data) {
	chomp;
	# process: folded_stack count
	# eg: func_a;func_b;func_c 31
	my ($stack, $samples) = (/^(.*)\s+?(\d+(?:\.\d*)?)$/);
	unless (defined $samples and defined $stack) {
		++$ignored;
		next;
	}

	# there may be an extra samples column for differentials:
	my $samples2 = undef;
	if ($stack =~ /^(.*)\s+?(\d+(?:\.\d*)?)$/) {
		$samples2 = $samples;
		($stack, $samples) = $stack =~ (/^(.*)\s+?(\d+(?:\.\d*)?)$/);
	}
	$delta = undef;
	if (defined $samples2) {
		$delta = $samples2 - $samples;
		$maxdelta = abs($delta) if abs($delta) > $maxdelta;
	}

	# for chain graphs, annotate waker frames with "_[w]", for later
	# coloring. This is a hack, but has a precedent ("_[k]" from perf).
	if ($colors eq "chain") {
		my @parts = split ";--;", $stack;
		my @newparts = ();
		$stack = shift @parts;
		$stack .= ";--;";
		foreach my $part (@parts) {
			$part =~ s/;/_[w];/g;
			$part .= "_[w]";
			push @newparts, $part;
		}
		$stack .= join ";--;", @parts;
	}

	# merge frames and populate %Node:
	$last = flow($last, [ '', split ";", $stack ], $time, $delta);

	if (defined $samples2) {
		$time += $samples2;
	} else {
		$time += $samples;
	}
}
flow($last, [], $time, $delta);

warn "Ignored $ignored lines with invalid format\n" if $ignored;
unless ($time) {
	warn "ERROR: No stack counts found\n";
	my $im = SVG->new();
	# emit an error message SVG, for tools automating flamegraph use
	my $imageheight = $fontsize * 5;
	$im->header($imagewidth, $imageheight);
	$im->stringTTF($im->colorAllocate(0, 0, 0), $fonttype, $fontsize + 2,
	    0.0, int($imagewidth / 2), $fontsize * 2,
	    "ERROR: No valid input provided to flamegraph.pl.", "middle");
	print $im->svg;
	exit 2;
}
if ($timemax and $timemax < $time) {
	warn "Specified --total $timemax is less than actual total $time, so ignored\n"
	if $timemax/$time > 0.02; # only warn is significant (e.g., not rounding etc)
	undef $timemax;
}
$timemax ||= $time;

my $widthpertime = ($imagewidth - 2 * $xpad) / $timemax;
my $minwidth_time = $minwidth / $widthpertime;

# prune blocks that are too narrow and determine max depth
while (my ($id, $node) = each %Node) {
	my ($func, $depth, $etime) = split ";", $id;
	my $stime = $node->{stime};
	die "missing start for $id" if not defined $stime;

	if (($etime-$stime) < $minwidth_time) {
		delete $Node{$id};
		next;
	}
	$depthmax = $depth if $depth > $depthmax;
}

# draw canvas, and embed interactive JavaScript program
my $imageheight = (($depthmax + 1) * $frameheight) + $ypad1 + $ypad2;
$imageheight += $ypad3 if $subtitletext ne "";
my $im = SVG->new();
$im->header($imagewidth, $imageheight);
my $inc = <<INC;
<defs >
	<linearGradient id="background" y1="0" y2="1" x1="0" x2="0" >
		<stop stop-color="$bgcolor1" offset="5%" />
		<stop stop-color="$bgcolor2" offset="95%" />
	</linearGradient>
</defs>
<style type="text/css">
	.func_g:hover { stroke:black; stroke-width:0.5; cursor:pointer; }
</style>
<script type="text/ecmascript">
<![CDATA[
	var details, searchbtn, matchedtxt, svg;
	function init(evt) {
		details = document.getElementById("details").firstChild;
		searchbtn = document.getElementById("search");
		matchedtxt = document.getElementById("matched");
		svg = document.getElementsByTagName("svg")[0];
		searching = 0;
	}

	// mouse-over for info
	function s(node) {		// show
		info = g_to_text(node);
		details.nodeValue = "$nametype " + info;
	}
	function c() {			// clear
		details.nodeValue = ' ';
	}

	// ctrl-F for search
	window.addEventListener("keydown",function (e) {
		if (e.keyCode === 114 || (e.ctrlKey && e.keyCode === 70)) {
			e.preventDefault();
			search_prompt();
		}
	})

	// functions
	function find_child(parent, name, attr) {
		var children = parent.childNodes;
		for (var i=0; i<children.length;i++) {
			if (children[i].tagName == name)
				return (attr != undefined) ? children[i].attributes[attr].value : children[i];
		}
		return;
	}
	function orig_save(e, attr, val) {
		if (e.attributes["_orig_"+attr] != undefined) return;
		if (e.attributes[attr] == undefined) return;
		if (val == undefined) val = e.attributes[attr].value;
		e.setAttribute("_orig_"+attr, val);
	}
	function orig_load(e, attr) {
		if (e.attributes["_orig_"+attr] == undefined) return;
		e.attributes[attr].value = e.attributes["_orig_"+attr].value;
		e.removeAttribute("_orig_"+attr);
	}
	function g_to_text(e) {
		var text = find_child(e, "title").firstChild.nodeValue;
		return (text)
	}
	function g_to_func(e) {
		var func = g_to_text(e);
		// if there's any manipulation we want to do to the function
		// name before it's searched, do it here before returning.
		return (func);
	}
	function update_text(e) {
		var r = find_child(e, "rect");
		var t = find_child(e, "text");
		var w = parseFloat(r.attributes["width"].value) -3;
		var txt = find_child(e, "title").textContent.replace(/\\([^(]*\\)\$/,"");
		t.attributes["x"].value = parseFloat(r.attributes["x"].value) +3;

		// Smaller than this size won't fit anything
		if (w < 2*$fontsize*$fontwidth) {
			t.textContent = "";
			return;
		}

		t.textContent = txt;
		// Fit in full text width
		if (/^ *\$/.test(txt) || t.getSubStringLength(0, txt.length) < w)
			return;

		for (var x=txt.length-2; x>0; x--) {
			if (t.getSubStringLength(0, x+2) <= w) {
				t.textContent = txt.substring(0,x) + "..";
				return;
			}
		}
		t.textContent = "";
	}

	// zoom
	function zoom_reset(e) {
		if (e.attributes != undefined) {
			orig_load(e, "x");
			orig_load(e, "width");
		}
		if (e.childNodes == undefined) return;
		for(var i=0, c=e.childNodes; i<c.length; i++) {
			zoom_reset(c[i]);
		}
	}
	function zoom_child(e, x, ratio) {
		if (e.attributes != undefined) {
			if (e.attributes["x"] != undefined) {
				orig_save(e, "x");
				e.attributes["x"].value = (parseFloat(e.attributes["x"].value) - x - $xpad) * ratio + $xpad;
				if(e.tagName == "text") e.attributes["x"].value = find_child(e.parentNode, "rect", "x") + 3;
			}
			if (e.attributes["width"] != undefined) {
				orig_save(e, "width");
				e.attributes["width"].value = parseFloat(e.attributes["width"].value) * ratio;
			}
		}

		if (e.childNodes == undefined) return;
		for(var i=0, c=e.childNodes; i<c.length; i++) {
			zoom_child(c[i], x-$xpad, ratio);
		}
	}
	function zoom_parent(e) {
		if (e.attributes) {
			if (e.attributes["x"] != undefined) {
				orig_save(e, "x");
				e.attributes["x"].value = $xpad;
			}
			if (e.attributes["width"] != undefined) {
				orig_save(e, "width");
				e.attributes["width"].value = parseInt(svg.width.baseVal.value) - ($xpad*2);
			}
		}
		if (e.childNodes == undefined) return;
		for(var i=0, c=e.childNodes; i<c.length; i++) {
			zoom_parent(c[i]);
		}
	}
	function zoom(node) {
		var attr = find_child(node, "rect").attributes;
		var width = parseFloat(attr["width"].value);
		var xmin = parseFloat(attr["x"].value);
		var xmax = parseFloat(xmin + width);
		var ymin = parseFloat(attr["y"].value);
		var ratio = (svg.width.baseVal.value - 2*$xpad) / width;

		// XXX: Workaround for JavaScript float issues (fix me)
		var fudge = 0.0001;

		var unzoombtn = document.getElementById("unzoom");
		unzoombtn.style["opacity"] = "1.0";

		var el = document.getElementsByTagName("g");
		for(var i=0;i<el.length;i++){
			var e = el[i];
			var a = find_child(e, "rect").attributes;
			var ex = parseFloat(a["x"].value);
			var ew = parseFloat(a["width"].value);
			// Is it an ancestor
			if ($inverted == 0) {
				var upstack = parseFloat(a["y"].value) > ymin;
			} else {
				var upstack = parseFloat(a["y"].value) < ymin;
			}
			if (upstack) {
				// Direct ancestor
				if (ex <= xmin && (ex+ew+fudge) >= xmax) {
					e.style["opacity"] = "0.5";
					zoom_parent(e);
					e.onclick = function(e){unzoom(); zoom(this);};
					update_text(e);
				}
				// not in current path
				else
					e.style["display"] = "none";
			}
			// Children maybe
			else {
				// no common path
				if (ex < xmin || ex + fudge >= xmax) {
					e.style["display"] = "none";
				}
				else {
					zoom_child(e, xmin, ratio);
					e.onclick = function(e){zoom(this);};
					update_text(e);
				}
			}
		}
	}
	function unzoom() {
		var unzoombtn = document.getElementById("unzoom");
		unzoombtn.style["opacity"] = "0.0";

		var el = document.getElementsByTagName("g");
		for(i=0;i<el.length;i++) {
			el[i].style["display"] = "block";
			el[i].style["opacity"] = "1";
			zoom_reset(el[i]);
			update_text(el[i]);
		}
	}

	// search
	function reset_search() {
		var el = document.getElementsByTagName("rect");
		for (var i=0; i < el.length; i++) {
			orig_load(el[i], "fill")
		}
	}
	function search_prompt() {
		if (!searching) {
			var term = prompt("Enter a search term (regexp " +
			    "allowed, eg: ^ext4_)", "");
			if (term != null) {
				search(term)
			}
		} else {
			reset_search();
			searching = 0;
			searchbtn.style["opacity"] = "0.1";
			searchbtn.firstChild.nodeValue = "Search"
			matchedtxt.style["opacity"] = "0.0";
			matchedtxt.firstChild.nodeValue = ""
		}
	}
	function search(term) {
		var re = new RegExp(term);
		var el = document.getElementsByTagName("g");
		var matches = new Object();
		var maxwidth = 0;
		for (var i = 0; i < el.length; i++) {
			var e = el[i];
			if (e.attributes["class"].value != "func_g")
				continue;
			var func = g_to_func(e);
			var rect = find_child(e, "rect");
			if (rect == null) {
				// the rect might be wrapped in an anchor
				// if nameattr href is being used
				if (rect = find_child(e, "a")) {
				    rect = find_child(r, "rect");
				}
			}
			if (func == null || rect == null)
				continue;

			// Save max width. Only works as we have a root frame
			var w = parseFloat(rect.attributes["width"].value);
			if (w > maxwidth)
				maxwidth = w;

			if (func.match(re)) {
				// highlight
				var x = parseFloat(rect.attributes["x"].value);
				orig_save(rect, "fill");
				rect.attributes["fill"].value =
				    "$searchcolor";

				// remember matches
				if (matches[x] == undefined) {
					matches[x] = w;
				} else {
					if (w > matches[x]) {
						// overwrite with parent
						matches[x] = w;
					}
				}
				searching = 1;
			}
		}
		if (!searching)
			return;

		searchbtn.style["opacity"] = "1.0";
		searchbtn.firstChild.nodeValue = "Reset Search"

		// calculate percent matched, excluding vertical overlap
		var count = 0;
		var lastx = -1;
		var lastw = 0;
		var keys = Array();
		for (k in matches) {
			if (matches.hasOwnProperty(k))
				keys.push(k);
		}
		// sort the matched frames by their x location
		// ascending, then width descending
		keys.sort(function(a, b){
			return a - b;
		});
		// Step through frames saving only the biggest bottom-up frames
		// thanks to the sort order. This relies on the tree property
		// where children are always smaller than their parents.
		var fudge = 0.0001;	// JavaScript floating point
		for (var k in keys) {
			var x = parseFloat(keys[k]);
			var w = matches[keys[k]];
			if (x >= lastx + lastw - fudge) {
				count += w;
				lastx = x;
				lastw = w;
			}
		}
		// display matched percent
		matchedtxt.style["opacity"] = "1.0";
		pct = 100 * count / maxwidth;
		if (pct == 100)
			pct = "100"
		else
			pct = pct.toFixed(1)
		matchedtxt.firstChild.nodeValue = "Matched: " + pct + "%";
	}
	function searchover(e) {
		searchbtn.style["opacity"] = "1.0";
	}
	function searchout(e) {
		if (searching) {
			searchbtn.style["opacity"] = "1.0";
		} else {
			searchbtn.style["opacity"] = "0.1";
		}
	}
]]>
</script>
INC
$im->include($inc);
$im->filledRectangle(0, 0, $imagewidth, $imageheight, 'url(#background)');
my ($white, $black, $vvdgrey, $vdgrey, $dgrey) = (
	$im->colorAllocate(255, 255, 255),
	$im->colorAllocate(0, 0, 0),
	$im->colorAllocate(40, 40, 40),
	$im->colorAllocate(160, 160, 160),
	$im->colorAllocate(200, 200, 200),
    );
$im->stringTTF($black, $fonttype, $fontsize + 5, 0.0, int($imagewidth / 2), $fontsize * 2, $titletext, "middle");
if ($subtitletext ne "") {
	$im->stringTTF($vdgrey, $fonttype, $fontsize, 0.0, int($imagewidth / 2), $fontsize * 4, $subtitletext, "middle");
}
$im->stringTTF($black, $fonttype, $fontsize, 0.0, $xpad, $imageheight - ($ypad2 / 2), " ", "", 'id="details"');
$im->stringTTF($black, $fonttype, $fontsize, 0.0, $xpad, $fontsize * 2,
    "Reset Zoom", "", 'id="unzoom" onclick="unzoom()" style="opacity:0.0;cursor:pointer"');
$im->stringTTF($black, $fonttype, $fontsize, 0.0, $imagewidth - $xpad - 100,
    $fontsize * 2, "Search", "", 'id="search" onmouseover="searchover()" onmouseout="searchout()" onclick="search_prompt()" style="opacity:0.1;cursor:pointer"');
$im->stringTTF($black, $fonttype, $fontsize, 0.0, $imagewidth - $xpad - 100, $imageheight - ($ypad2 / 2), " ", "", 'id="matched"');

if ($palette) {
	read_palette();
}

# draw frames
while (my ($id, $node) = each %Node) {
	my ($func, $depth, $etime) = split ";", $id;
	my $stime = $node->{stime};
	my $delta = $node->{delta};

	$etime = $timemax if $func eq "" and $depth == 0;

	my $x1 = $xpad + $stime * $widthpertime;
	my $x2 = $xpad + $etime * $widthpertime;
	my ($y1, $y2);
	unless ($inverted) {
		$y1 = $imageheight - $ypad2 - ($depth + 1) * $frameheight + $framepad;
		$y2 = $imageheight - $ypad2 - $depth * $frameheight;
	} else {
		$y1 = $ypad1 + $depth * $frameheight;
		$y2 = $ypad1 + ($depth + 1) * $frameheight - $framepad;
	}

	my $samples = sprintf "%.0f", ($etime - $stime) * $factor;
	(my $samples_txt = $samples) # add commas per perlfaq5
		=~ s/(^[-+]?\d+?(?=(?>(?:\d{3})+)(?!\d))|\G\d{3}(?=\d))/$1,/g;

	my $info;
	if ($func eq "" and $depth == 0) {
		$info = "all ($samples_txt $countname, 100%)";
	} else {
		my $pct = sprintf "%.2f", ((100 * $samples) / ($timemax * $factor));
		my $escaped_func = $func;
		# clean up SVG breaking characters:
		$escaped_func =~ s/&/&amp;/g;
		$escaped_func =~ s/</&lt;/g;
		$escaped_func =~ s/>/&gt;/g;
		$escaped_func =~ s/"/&quot;/g;
		$escaped_func =~ s/_\[[kwij]\]$//;	# strip any annotation
		unless (defined $delta) {
			$info = "$escaped_func ($samples_txt $countname, $pct%)";
		} else {
			my $d = $negate ? -$delta : $delta;
			my $deltapct = sprintf "%.2f", ((100 * $d) / ($timemax * $factor));
			$deltapct = $d > 0 ? "+$deltapct" : $deltapct;
			$info = "$escaped_func ($samples_txt $countname, $pct%; $deltapct%)";
		}
	}

	my $nameattr = { %{ $nameattr{$func}||{} } }; # shallow clone
	$nameattr->{class}       ||= "func_g";
	$nameattr->{onmouseover} ||= "s(this)";
	$nameattr->{onmouseout}  ||= "c()";
	$nameattr->{onclick}     ||= "zoom(this)";
	$nameattr->{title}       ||= $info;
	$im->group_start($nameattr);

	my $color;
	if ($func eq "--") {
		$color = $vdgrey;
	} elsif ($func eq "-") {
		$color = $dgrey;
	} elsif (defined $delta) {
		$color = color_scale($delta, $maxdelta);
	} elsif ($palette) {
		$color = color_map($colors, $func);
	} else {
		$color = color($colors, $hash, $func);
	}
	$im->filledRectangle($x1, $y1, $x2, $y2, $color, 'rx="2" ry="2"');

	my $chars = int( ($x2 - $x1) / ($fontsize * $fontwidth));
	my $text = "";
	if ($chars >= 3) { # room for one char plus two dots
		$func =~ s/_\[[kwij]\]$//;	# strip any annotation
		$text = substr $func, 0, $chars;
		substr($text, -2, 2) = ".." if $chars < length $func;
		$text =~ s/&/&amp;/g;
		$text =~ s/</&lt;/g;
		$text =~ s/>/&gt;/g;
	}
	$im->stringTTF($black, $fonttype, $fontsize, 0.0, $x1 + 3, 3 + ($y1 + $y2) / 2, $text, "");

	$im->group_end($nameattr);
}

print $im->svg;

if ($palette) {
	write_palette();
}

# vim: ts=8 sts=8 sw=8 noexpandtab
