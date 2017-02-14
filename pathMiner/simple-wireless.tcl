#Copyright (c) 1997 Regents of the University of California.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. All advertising materials mentioning features or use of this software
#    must display the following acknowledgement:
#      This product includes software developed by the Computer Systems
#      Engineering Group at Lawrence Berkeley Laboratory.
# 4. Neither the name of the University nor of the Laboratory may be used
#    to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#
# simple-wireless.tcl
# A simple example for wireless simulation

# ======================================================================
# Define options
# ======================================================================
set val(chan)           Channel/WirelessChannel    ;# channel type
set val(prop)           Propagation/TwoRayGround   ;# radio-propagation model
set val(netif)          Phy/WirelessPhy            ;# network interface type
set val(mac)            Mac/802_11                 ;# MAC type
set val(ifq)            Queue/DropTail/PriQueue    ;# interface queue type
set val(ll)             LL                         ;# link layer type
set val(ant)            Antenna/OmniAntenna        ;# antenna model
set val(ifqlen)         50                         ;# max packet in ifq
set val(nn)             5                          ;# number of mobilenodes
set val(rp)             DSDV                       ;# routing protocol
set val(x) 		 500
set val(y)		 500
set rng_time [expr rand() * 150]
# ======================================================================
# Main Program
# ======================================================================


#
# Initialize Global Variables
#
set ns_		[new Simulator]
set tracefd     [open simple-wireless.tr w]
set nf  [open simple-wireless.nam w]
$ns_ trace-all $tracefd
$ns_ namtrace-all-wireless $nf $val(x) $val(y)

# set up topography object
set topo       [new Topography]

$topo load_flatgrid 500 500

#
# Create God
#
create-god $val(nn)

#
#  Create the specified number of mobilenodes [$val(nn)] and "attach" them
#  to the channel. 
#  Here two nodes are created : node(0) and node(1)

# configure node

        $ns_ node-config -adhocRouting $val(rp) \
			 -llType $val(ll) \
			 -macType $val(mac) \
			 -ifqType $val(ifq) \
			 -ifqLen $val(ifqlen) \
			 -antType $val(ant) \
			 -propType $val(prop) \
			 -phyType $val(netif) \
			 -channelType $val(chan) \
			 -topoInstance $topo \
			 -agentTrace ON \
			 -routerTrace ON \
			 -macTrace OFF \
			 -movementTrace OFF			
			 
	for {set i 0} {$i < $val(nn) } {incr i} {
		set node_($i) [$ns_ node]
		$node_($i) random-motion 0 ;	
	}


#
# Provide initial (X,Y, for now Z=0) co-ordinates for mobilenodes
#
$node_(0) set X_ 5.0
$node_(0) set Y_ 2.0
$node_(0) set Z_ 0.0

$node_(1) set X_ 390.0
$node_(1) set Y_ 385.0
$node_(1) set Z_ 0.0

$node_(2) set X_ 190.0
$node_(2) set Y_ 85.0
$node_(2) set Z_ 0.0

$node_(3) set X_ 380.0
$node_(3) set Y_ 5.0
$node_(3) set Z_ 0.0

$node_(4) set X_ 10.0
$node_(4) set Y_ 385.0
$node_(4) set Z_ 0.0


#enable node trace in namfor {set i 0} {$i < $val(nn)} {incr i} {	$node_($i) namattach $nf	$ns_ initial_node_pos $node_($i) 20}

#
# Now produce some simple node movements

#set xx_ [expr rand()*$val(x)]
#set yy_ [expr rand()*$val(y)]
#$ns_ at $rng_time "$node_(1) setdest $xx_ $yy_ 15.0"   ;# random movements

$ns_ at 00.0 "$node_(1) setdest 25.0 20.0 15.0" 
$ns_ at 100.0 "$node_(1) setdest 490.0 480.0 15.0"

# Setup traffic flow between nodes
# TCP connections between node_(0) and node_(1)

set tcp0 [new Agent/TCP]
$tcp0 set class_ 2
set tcp2 [new Agent/TCP]
$tcp2 set class_ 2
set tcp3 [new Agent/TCP]
$tcp3 set class_ 2
set tcp4 [new Agent/TCP]
$tcp4 set class_ 2
set sink [new Agent/TCPSink]
$ns_ attach-agent $node_(0) $tcp0
$ns_ attach-agent $node_(2) $tcp2
$ns_ attach-agent $node_(3) $tcp3
$ns_ attach-agent $node_(4) $tcp4
$ns_ attach-agent $node_(1) $sink
$ns_ connect $tcp0 $sink
$ns_ connect $tcp2 $sink
$ns_ connect $tcp3 $sink
$ns_ connect $tcp4 $sink
set ftp0 [new Application/FTP]
$ftp0 attach-agent $tcp0
$ns_ at 10.0 "$ftp0 start" 
set ftp2 [new Application/FTP]
$ftp2 attach-agent $tcp2
$ns_ at 10.0 "$ftp2 start" 
set ftp3 [new Application/FTP]
$ftp3 attach-agent $tcp3
$ns_ at 10.0 "$ftp3 start"
set ftp4 [new Application/FTP]
$ftp4 attach-agent $tcp4
$ns_ at 10.0 "$ftp4 start"
#
# Tell nodes when the simulation ends
#
for {set i 0} {$i < $val(nn) } {incr i} {
    $ns_ at 150.0 "$node_($i) reset";
}
$ns_ at 0.0 "$node_(1) label \"Person\""
$ns_ at 150.0 "stop"
$ns_ at 150.01 "puts \"NS EXITING...\" ; $ns_ halt"
proc stop {} {
    global ns_ tracefd
    $ns_ flush-trace
    close $tracefd
}

puts "Starting Simulation..."
$ns_ run

