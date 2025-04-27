import React, { useEffect, useRef } from 'react';
import { Paper, Typography } from '@material-ui/core';
import { Terminal } from 'xterm';
import { FitAddon } from 'xterm-addon-fit';
import 'xterm/css/xterm.css';

function XTermShell({ dataChannel, dataChannelReady, shellResponses }) {
  const xtermRef = useRef(null);
  const fitAddon = useRef(null);
  const term = useRef(null);
  const lastShellLength = useRef(0);

  // Initialize xterm.js
  useEffect(() => {
    if (!xtermRef.current) return;
    if (!term.current) {
      term.current = new Terminal({
        fontFamily: 'monospace',
        fontSize: 14,
        theme: {
          background: '#181818',
          foreground: '#f0f0f0',
        },
        cursorBlink: true,
        scrollback: 1000,
      });
      fitAddon.current = new FitAddon();
      term.current.loadAddon(fitAddon.current);
      term.current.open(xtermRef.current);
      fitAddon.current.fit();

      // Send user input to backend as PTY data
      term.current.onData(data => {
        if (dataChannelReady && dataChannel) {
          dataChannel.send(JSON.stringify({ action: "shell_input", data }));
        }
      });
    }
    fitAddon.current.fit();
    // eslint-disable-next-line
  }, [xtermRef, dataChannel, dataChannelReady]);

  // Write new shell output to terminal
  useEffect(() => {
    if (!term.current) return;
    const allOutput = shellResponses.map(r => r.output).join('');
    if (allOutput.length > lastShellLength.current) {
      const newOutput = allOutput.slice(lastShellLength.current);
      term.current.write(newOutput);
      lastShellLength.current = allOutput.length;
    }
  }, [shellResponses]);

  // Fit terminal on window resize
  useEffect(() => {
    const handleResize = () => {
      if (fitAddon.current) fitAddon.current.fit();
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <Paper style={{ marginTop: 20, padding: 0, background: '#181818', color: '#f0f0f0', borderRadius: 10 }} elevation={3}>
      <div ref={xtermRef} style={{ width: '100%', height: 320, minHeight: 200 }} />
    </Paper>
  );
}

export default XTermShell;
