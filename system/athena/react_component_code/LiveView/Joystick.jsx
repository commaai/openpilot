import React, { useState, useEffect, useRef } from "react";
import {
  Paper,
  Typography,
  FormControlLabel,
  Switch,
  IconButton,
  Button,
} from "@material-ui/core";
import FullscreenIcon from "@material-ui/icons/Fullscreen";
import CloseIcon from "@material-ui/icons/Close";
import Colors from '../../colors';

function Joystick({
  dataChannel,
  dataChannelReady,
  controllerEnabled,
  setControllerEnabled,
  updateControllerState,
  useVirtualControls,
  setUseVirtualControls,
}) {
  // ----- State Hooks -----
  const [controllerStatus, setControllerStatus] = useState("No controller detected");
  const [controllerConnected, setControllerConnected] = useState(false);
  const [lastInputs, setLastInputs] = useState({});
  const [throttle, setThrottle] = useState(0);
  const [steering, setSteering] = useState(0);
  const [isMobile, setIsMobile] = useState(false);
  const [invertSteering, setInvertSteering] = useState(false);
  const [fullscreenMode, setFullscreenMode] = useState(false);
  const [touchState, setTouchState] = useState({
    joystickTouch: null,
    currentSteering: 0,
    currentThrottle: 0,
    joystickCenter: { x: 0, y: 0 },
    mouseDown: false,
    joystickRect: null,
  });

  // ----- Refs -----
  const virtualControllerRef = useRef(null);
  const requestRef = useRef();
  const previousTimeRef = useRef();

  // ----- Utility: Safe Setter for Virtual Controls -----
  const safeSetUseVirtualControls = setUseVirtualControls || (() => console.warn("setUseVirtualControls is not available"));

  // ----- Effect: Mobile Device Check -----
  useEffect(() => {
    const checkIfMobile = () => {
      const userAgent = navigator.userAgent || navigator.vendor || window.opera;
      const mobileRegex = /android|iPad|iPhone|iPod|webOS|BlackBerry|Windows Phone/i;
      return mobileRegex.test(userAgent);
    };
    const isMobileDevice = checkIfMobile();
    setIsMobile(isMobileDevice);
    // Optionally: safeSetUseVirtualControls(true);
  }, [safeSetUseVirtualControls]);

  // ----- Function: Check for Physical Controller -----
  const checkForController = () => {
    if (useVirtualControls) {
      setControllerConnected(true);
      setControllerStatus("Using virtual controls");
      return true;
    }
    const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
    for (let i = 0; i < gamepads.length; i++) {
      if (
        gamepads[i] &&
        (gamepads[i].id.toLowerCase().includes("xbox") ||
          gamepads[i].id.toLowerCase().includes("xinput") ||
          gamepads[i].id.toLowerCase().includes("gamepad"))
      ) {
        setControllerConnected(true);
        setControllerStatus(`Connected: ${gamepads[i].id}`);
        return true;
      }
    }
    setControllerConnected(false);
    setControllerStatus("No controller detected");
    return false;
  };

  // ----- Touch Handlers -----
  const handleJoystickTouchStart = (e) => {
    if (!controllerEnabled || !dataChannelReady) return;
    e.preventDefault();
    const touch = e.touches[0];
    const rect = e.currentTarget.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;
    setTouchState((prev) => ({
      ...prev,
      joystickTouch: {
        id: touch.identifier,
        startX: touch.clientX,
        startY: touch.clientY,
      },
      joystickCenter: { x: centerX, y: centerY },
    }));
  };

  const handleJoystickTouchMove = (e) => {
    if (!touchState.joystickTouch || !controllerEnabled) return;
    e.preventDefault();

    // Find the correct touch
    let touch = null;
    for (let i = 0; i < e.touches.length; i++) {
      if (e.touches[i].identifier === touchState.joystickTouch.id) {
        touch = e.touches[i];
        break;
      }
    }
    if (!touch) return;

    const rect = e.currentTarget.getBoundingClientRect();
    const maxHorizontalOffset = rect.width / 2;
    const horizontalOffset = touch.clientX - touchState.joystickCenter.x;
    let newSteering = Math.max(-1, Math.min(1, horizontalOffset / maxHorizontalOffset));
    newSteering = Math.round(newSteering * 100) / 100;

    const maxVerticalOffset = rect.height / 2;
    const verticalOffset = touchState.joystickCenter.y - touch.clientY;
    let newThrottle = Math.max(-1, Math.min(1, verticalOffset / maxVerticalOffset));
    newThrottle = Math.round(newThrottle * 100) / 100;

    setTouchState((prev) => ({
      ...prev,
      currentSteering: newSteering,
      currentThrottle: newThrottle,
    }));
    setSteering(newSteering);
    setThrottle(newThrottle);
  };

  const handleJoystickTouchEnd = (e) => {
    e.preventDefault();
    const remainingTouchIds = Array.from(e.touches).map((t) => t.identifier);
    if (touchState.joystickTouch && !remainingTouchIds.includes(touchState.joystickTouch.id)) {
      // Reset to neutral position
      setTouchState((prev) => ({
        ...prev,
        joystickTouch: null,
        currentSteering: 0,
        currentThrottle: 0,
      }));
      setSteering(0);
      setThrottle(0);
    }
  };

  // ----- Mouse Handlers (Desktop) -----
  const handleJoystickMouseDown = (e) => {
    if (!controllerEnabled || !dataChannelReady) return;
    e.preventDefault();
    const rect = e.currentTarget.getBoundingClientRect();
    const centerX = rect.left + rect.width / 2;
    const centerY = rect.top + rect.height / 2;

    const maxHorizontalOffset = rect.width / 2;
    const horizontalOffset = e.clientX - centerX;
    let newSteering = Math.max(-1, Math.min(1, horizontalOffset / maxHorizontalOffset));
    newSteering = Math.round(newSteering * 100) / 100;

    const maxVerticalOffset = rect.height / 2;
    const verticalOffset = centerY - e.clientY;
    let newThrottle = Math.max(-1, Math.min(1, verticalOffset / maxVerticalOffset));
    newThrottle = Math.round(newThrottle * 100) / 100;

    setSteering(newSteering);
    setThrottle(newThrottle);
    setTouchState((prev) => ({
      ...prev,
      mouseDown: true,
      joystickCenter: { x: centerX, y: centerY },
      joystickRect: rect,
      currentSteering: newSteering,
      currentThrottle: newThrottle,
    }));
  };

  const handleJoystickMouseMove = (e) => {
    if (!touchState.mouseDown || !controllerEnabled || !touchState.joystickRect) return;
    e.preventDefault();

    const rect = touchState.joystickRect;
    const maxHorizontalOffset = rect.width / 2;
    const horizontalOffset = e.clientX - touchState.joystickCenter.x;
    let newSteering = Math.max(-1, Math.min(1, horizontalOffset / maxHorizontalOffset));
    newSteering = Math.round(newSteering * 100) / 100;

    const maxVerticalOffset = rect.height / 2;
    const verticalOffset = touchState.joystickCenter.y - e.clientY;
    let newThrottle = Math.max(-1, Math.min(1, verticalOffset / maxVerticalOffset));
    newThrottle = Math.round(newThrottle * 100) / 100;

    const threshold = 0.01;
    if (
      Math.abs(touchState.currentSteering - newSteering) > threshold ||
      Math.abs(touchState.currentThrottle - newThrottle) > threshold
    ) {
      setSteering(newSteering);
      setThrottle(newThrottle);
      setTouchState((prev) => ({
        ...prev,
        currentSteering: newSteering,
        currentThrottle: newThrottle,
      }));
    }
  };

  const handleJoystickMouseUp = (e) => {
    e.preventDefault();
    if (touchState.mouseDown) {
      setTouchState((prev) => ({
        ...prev,
        mouseDown: false,
        currentSteering: 0,
        currentThrottle: 0,
      }));
      setSteering(0);
      setThrottle(0);
    }
  };

  // ----- Global Mouse & Blur Handlers for Desktop -----
  useEffect(() => {
    if (useVirtualControls) {
      document.addEventListener("mousemove", handleJoystickMouseMove);
      document.addEventListener("mouseup", handleJoystickMouseUp);

      const handleBlur = () => {
        if (touchState.mouseDown) {
          setTouchState((prev) => ({
            ...prev,
            mouseDown: false,
            currentSteering: 0,
            currentThrottle: 0,
          }));
          setSteering(0);
          setThrottle(0);
        }
      };
      window.addEventListener("blur", handleBlur);

      return () => {
        document.removeEventListener("mousemove", handleJoystickMouseMove);
        document.removeEventListener("mouseup", handleJoystickMouseUp);
        window.removeEventListener("blur", handleBlur);
      };
    }
  }, [useVirtualControls, touchState.mouseDown, controllerEnabled]);

  // ----- Prevent Default Controller Navigation -----
  useEffect(() => {
    const preventDefaultForGamepad = (e) => {
      if (e.sourceCapabilities && e.sourceCapabilities.firesTouchEvents === false) {
        e.preventDefault();
        e.stopPropagation();
      }
    };

    window.addEventListener("gamepadconnected", (e) => {
      console.log("Gamepad connected:", e.gamepad.id);
    });
    window.addEventListener("popstate", preventDefaultForGamepad);
    document.addEventListener(
      "keydown",
      (e) => {
        if (controllerEnabled && !useVirtualControls) {
          if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Backspace", "Escape"].includes(e.key)) {
            e.preventDefault();
          }
        }
      },
      { passive: false }
    );
    window.addEventListener("gamepadButtonDown", preventDefaultForGamepad, { passive: false });

    return () => {
      window.removeEventListener("popstate", preventDefaultForGamepad);
      window.removeEventListener("gamepadButtonDown", preventDefaultForGamepad);
      document.removeEventListener("keydown", preventDefaultForGamepad);
    };
  }, [controllerEnabled, useVirtualControls]);

  // ----- Keyboard Controls for Accessibility -----
  useEffect(() => {
    if (!useVirtualControls || !controllerEnabled || !dataChannelReady) return;

    const keyValues = {
      ArrowUp: { throttle: 1, steering: 0 },
      ArrowDown: { throttle: -1, steering: 0 },
      ArrowLeft: { throttle: 0, steering: -1 },
      ArrowRight: { throttle: 0, steering: 1 },
      KeyW: { throttle: 1, steering: 0 },
      KeyS: { throttle: -1, steering: 0 },
      KeyA: { throttle: 0, steering: -1 },
      KeyD: { throttle: 0, steering: 1 },
    };

    const pressedKeys = {
      ArrowUp: false,
      ArrowDown: false,
      ArrowLeft: false,
      ArrowRight: false,
      KeyW: false,
      KeyS: false,
      KeyA: false,
      KeyD: false,
    };

    const updateControlsFromKeys = () => {
      let newThrottle = 0;
      let newSteering = 0;
      if ((pressedKeys.ArrowUp || pressedKeys.KeyW) && !(pressedKeys.ArrowDown || pressedKeys.KeyS)) {
        newThrottle = 1;
      } else if ((pressedKeys.ArrowDown || pressedKeys.KeyS) && !(pressedKeys.ArrowUp || pressedKeys.KeyW)) {
        newThrottle = -1;
      }
      if ((pressedKeys.ArrowLeft || pressedKeys.KeyA) && !(pressedKeys.ArrowRight || pressedKeys.KeyD)) {
        newSteering = -1;
      } else if ((pressedKeys.ArrowRight || pressedKeys.KeyD) && !(pressedKeys.ArrowLeft || pressedKeys.KeyA)) {
        newSteering = 1;
      }
      setSteering(newSteering);
      setThrottle(newThrottle);
      setTouchState((prev) => ({
        ...prev,
        currentSteering: newSteering,
        currentThrottle: newThrottle,
      }));
    };

    const handleKeyDown = (e) => {
      const keyCode = e.code || e.key;
      if (keyValues[keyCode] && !pressedKeys[keyCode]) {
        e.preventDefault();
        pressedKeys[keyCode] = true;
        updateControlsFromKeys();
      }
    };

    const handleKeyUp = (e) => {
      const keyCode = e.code || e.key;
      if (keyValues[keyCode]) {
        e.preventDefault();
        pressedKeys[keyCode] = false;
        updateControlsFromKeys();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    document.addEventListener("keyup", handleKeyUp);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.removeEventListener("keyup", handleKeyUp);
    };
  }, [useVirtualControls, controllerEnabled, dataChannelReady, invertSteering]);

  // ----- Effect: Update Controller State -----
  useEffect(() => {
    if (updateControllerState) {
      updateControllerState(steering, throttle, invertSteering);
    }
  }, [steering, throttle, invertSteering, updateControllerState]);

  // ----- Function: Send Control Inputs to Server -----
  const sendControlInputs = (steeringValue, throttleValue) => {
    if (!controllerEnabled || !dataChannelReady || !dataChannel) return;
    steeringValue = Math.round(steeringValue * 100) / 100;
    throttleValue = Math.round(throttleValue * 100) / 100;
    const finalSteeringValue = invertSteering ? -steeringValue : steeringValue;

    const controlData = {
      action: "controller",
      throttle: throttleValue,
      steering: finalSteeringValue,
      timestamp: Date.now(),
    };

    try {
      dataChannel.send(JSON.stringify(controlData));
      setLastInputs({
        throttle: throttleValue,
        steering: finalSteeringValue,
        displaySteering: steeringValue,
      });
    } catch (err) {
      console.error("Error sending controller data:", err);
    }

    if (updateControllerState) {
      updateControllerState(steeringValue, throttleValue, invertSteering);
    }
  };

  // ----- Function: Handle Physical Controller Input -----
  const handleControllerInput = () => {
    if (!controllerEnabled || !dataChannelReady || !dataChannel || useVirtualControls) return;
    const gamepads = navigator.getGamepads ? navigator.getGamepads() : [];
    let controllerFound = false;

    for (let i = 0; i < gamepads.length; i++) {
      if (!gamepads[i]) continue;
      controllerFound = true;
      const gamepad = gamepads[i];

      let steeringValue = gamepad.axes[0];
      if (Math.abs(steeringValue) < 0.1) steeringValue = 0;

      let throttleValue = 0;
      if (gamepad.buttons && gamepad.buttons.length > 7) {
        throttleValue = gamepad.buttons[7].value || 0;
      }

      let brakeValue = 0;
      if (gamepad.buttons && gamepad.buttons.length > 6) {
        brakeValue = gamepad.buttons[6].value || 0;
      }

      const finalThrottle = throttleValue - brakeValue;
      setSteering(steeringValue);
      setThrottle(finalThrottle);
      sendControlInputs(steeringValue, finalThrottle);
      break;
    }

    if (!controllerFound) {
      sendControlInputs(0, 0);
    }

    if (controllerConnected !== controllerFound && !useVirtualControls) {
      setControllerConnected(controllerFound);
      checkForController();
      if (!controllerFound && isMobile) {
        setControllerStatus("Controller disconnected. Consider using virtual controls.");
      }
    }
  };

  // ----- Effect: Controller Detection & Input Polling -----
  useEffect(() => {
    checkForController();

    const handleControllerConnected = () => {
      console.log("Controller connected!");
      checkForController();
    };

    const handleControllerDisconnected = () => {
      console.log("Controller disconnected!");
      checkForController();
      if (isMobile) {
        setControllerStatus("Controller disconnected. Consider using virtual controls.");
      }
    };

    window.addEventListener("gamepadconnected", handleControllerConnected);
    window.addEventListener("gamepaddisconnected", handleControllerDisconnected);

    let inputInterval = null;
    const pollRate = 10; // 100Hz

    if (controllerEnabled) {
      if (useVirtualControls) {
        const animate = (time) => {
          if (previousTimeRef.current === undefined) {
            previousTimeRef.current = time;
          }
          const deltaTime = time - previousTimeRef.current;
          if (deltaTime >= 10) {
            previousTimeRef.current = time;
            sendControlInputs(touchState.currentSteering, touchState.currentThrottle);
          }
          requestRef.current = requestAnimationFrame(animate);
        };

        if (!isMobile) {
          requestRef.current = requestAnimationFrame(animate);
        } else {
          inputInterval = setInterval(() => {
            sendControlInputs(touchState.currentSteering, touchState.currentThrottle);
          }, pollRate);
        }
      } else {
        inputInterval = setInterval(() => {
          handleControllerInput();
        }, pollRate);
      }
    }

    return () => {
      window.removeEventListener("gamepadconnected", handleControllerConnected);
      window.removeEventListener("gamepaddisconnected", handleControllerDisconnected);
      if (inputInterval) clearInterval(inputInterval);
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [controllerEnabled, dataChannelReady, dataChannel, useVirtualControls, touchState, isMobile, updateControllerState, lastInputs]);

  // ----- Fullscreen Handlers -----
  const enterFullscreen = () => {
    setFullscreenMode(true);
    document.body.style.overflow = "hidden";
  };

  const exitFullscreen = () => {
    setFullscreenMode(false);
    document.body.style.overflow = "";
  };

  // ----- Render: Virtual Controller UI -----
  const renderVirtualController = () => {
    const joystickThumbStyle = {
      position: "absolute",
      left: `${50 + steering * 50}%`,
      top: `${50 - throttle * 50}%`,
      transform: "translate(-50%, -50%)",
      width: fullscreenMode ? "60px" : "40px",
      height: fullscreenMode ? "60px" : "40px",
      borderRadius: "50%",
      backgroundColor: Colors.blue500,
      border: "3px solid white",
      ...(isMobile && { transition: "left 0.1s ease-out, top 0.1s ease-out" }),
    };

    if (fullscreenMode) {
      return (
        <div
          ref={virtualControllerRef}
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            zIndex: 9999,
            backgroundColor: Colors.grey900,
            display: "flex",
            flexDirection: "column",
            paddingTop: "20px",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0 20px 20px 20px" }}>
            <Typography variant="h5" style={{ color: Colors.white }}>
              Virtual Controller {invertSteering && <span style={{ color: Colors.orange500 }}>(Steering Inverted)</span>}
            </Typography>
            <div>
              <FormControlLabel
                control={
                  <Switch
                    checked={invertSteering}
                    onChange={(e) => setInvertSteering(e.target.checked)}
                    color="secondary"
                    size="small"
                  />
                }
                label={<Typography style={{ color: Colors.white }}>Invert Steering</Typography>}
              />
              <IconButton onClick={exitFullscreen} style={{ color: Colors.white }}>
                <CloseIcon />
              </IconButton>
            </div>
          </div>

          <div
            style={{
              flex: 1,
              backgroundColor: Colors.grey800,
              position: "relative",
              touchAction: "none",
              cursor: "grab",
            }}
            onTouchStart={handleJoystickTouchStart}
            onTouchMove={handleJoystickTouchMove}
            onTouchEnd={handleJoystickTouchEnd}
            onTouchCancel={handleJoystickTouchEnd}
            onMouseDown={handleJoystickMouseDown}
          >
            <div
              style={{
                position: "absolute",
                left: "50%",
                top: 0,
                bottom: 0,
                width: "2px",
                backgroundColor: Colors.white30,
              }}
            />
            <div
              style={{
                position: "absolute",
                top: "50%",
                left: 0,
                right: 0,
                height: "2px",
                backgroundColor: Colors.white30,
              }}
            />
            <div style={joystickThumbStyle} />
            <div
              style={{
                position: "absolute",
                left: "20px",
                top: "50%",
                transform: "translateY(-50%)",
                color: Colors.white60,
                fontSize: "24px",
              }}
            >
              {invertSteering ? "RIGHT" : "LEFT"}
            </div>
            <div
              style={{
                position: "absolute",
                right: "20px",
                top: "50%",
                transform: "translateY(-50%)",
                color: Colors.white60,
                fontSize: "24px",
              }}
            >
              {invertSteering ? "LEFT" : "RIGHT"}
            </div>
            <div
              style={{
                position: "absolute",
                left: "50%",
                top: "20px",
                transform: "translateX(-50%)",
                color: Colors.green500,
                fontSize: "24px",
              }}
            >
              THROTTLE
            </div>
            <div
              style={{
                position: "absolute",
                left: "50%",
                bottom: "20px",
                transform: "translateX(-50%)",
                color: Colors.orange500,
                fontSize: "24px",
              }}
            >
              BRAKE
            </div>
          </div>

          <div style={{ display: "flex", justifyContent: "space-between", padding: "20px" }}>
            <Typography style={{ color: Colors.white, fontSize: "18px" }}>
              Steering: {steering.toFixed(2)}
            </Typography>
            <Typography style={{ color: Colors.white, fontSize: "18px" }}>
              Throttle: {throttle.toFixed(2)}
            </Typography>
          </div>
        </div>
      );
    }

    return (
      <div style={{ marginTop: "15px" }} ref={virtualControllerRef}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
          <Typography style={{ color: Colors.white, textAlign: "center" }}>
            Virtual Controller {invertSteering && <span style={{ color: Colors.orange500 }}>(Steering Inverted)</span>}
          </Typography>
          <div style={{ display: "flex", alignItems: "center" }}>
            <FormControlLabel
              control={
                <Switch
                  checked={invertSteering}
                  onChange={(e) => setInvertSteering(e.target.checked)}
                  color="secondary"
                  size="small"
                />
              }
              label={<Typography style={{ color: Colors.white }}>Invert</Typography>}
            />
            <IconButton onClick={enterFullscreen} style={{ color: Colors.white }} size="small">
              <FullscreenIcon />
            </IconButton>
          </div>
        </div>
        <div
          style={{
            width: "100%",
            height: "200px",
            backgroundColor: Colors.grey900,
            borderRadius: "10px",
            position: "relative",
            touchAction: "none",
            marginBottom: "10px",
            cursor: "grab",
          }}
          onTouchStart={handleJoystickTouchStart}
          onTouchMove={handleJoystickTouchMove}
          onTouchEnd={handleJoystickTouchEnd}
          onTouchCancel={handleJoystickTouchEnd}
          onMouseDown={handleJoystickMouseDown}
        >
          <div
            style={{
              position: "absolute",
              left: "50%",
              top: 0,
              bottom: 0,
              width: "2px",
              backgroundColor: Colors.white30,
            }}
          />
          <div
            style={{
              position: "absolute",
              top: "50%",
              left: 0,
              right: 0,
              height: "2px",
              backgroundColor: Colors.white30,
            }}
          />
          <div style={joystickThumbStyle} />
          <div
            style={{
              position: "absolute",
              left: "10px",
              top: "50%",
              transform: "translateY(-50%)",
              color: Colors.white60,
            }}
          >
            {invertSteering ? "RIGHT" : "LEFT"}
          </div>
          <div
            style={{
              position: "absolute",
              right: "10px",
              top: "50%",
              transform: "translateY(-50%)",
              color: Colors.white60,
            }}
          >
            {invertSteering ? "LEFT" : "RIGHT"}
          </div>
          <div
            style={{
              position: "absolute",
              left: "50%",
              top: "10px",
              transform: "translateX(-50%)",
              color: Colors.green500,
            }}
          >
            THROTTLE
          </div>
          <div
            style={{
              position: "absolute",
              left: "50%",
              bottom: "10px",
              transform: "translateX(-50%)",
              color: Colors.orange500,
            }}
          >
            BRAKE
          </div>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <Typography style={{ color: Colors.white }}>
            Steering: {steering.toFixed(2)}
          </Typography>
          <Typography style={{ color: Colors.white }}>
            Throttle: {throttle.toFixed(2)}
          </Typography>
        </div>
        <Typography style={{ color: Colors.white60, fontSize: "0.8rem", textAlign: "center", marginTop: "5px" }}>
          {isMobile
            ? "Touch and drag to control. Left/right = steering, up/down = throttle/brake"
            : "Click and drag to control. Left/right = steering, up/down = throttle/brake"}
        </Typography>
        {!isMobile && (
          <Typography style={{ color: Colors.white60, fontSize: "0.8rem", textAlign: "center", marginTop: "5px" }}>
            Keyboard: Arrow keys or WASD for steering and throttle
          </Typography>
        )}
      </div>
    );
  };

  // ----- Main Render -----
  return (
    <Paper
      elevation={3}
      style={{
        marginTop: "20px",
        padding: "15px",
        backgroundColor: Colors.grey800,
        borderRadius: "10px",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "10px" }}>
        <Typography variant="h6" style={{ color: Colors.white }}>
          {useVirtualControls ? "Virtual Controller" : "Xbox Controller"}
        </Typography>
        <div style={{ display: "flex", alignItems: "center" }}>
          <FormControlLabel
            control={
              <Switch
                checked={useVirtualControls}
                onChange={(e) => safeSetUseVirtualControls(e.target.checked)}
                color="secondary"
              />
            }
            label={<Typography style={{ color: Colors.white, marginRight: "15px" }}>Virtual</Typography>}
          />
          {!useVirtualControls && (
            <FormControlLabel
              control={
                <Switch
                  checked={invertSteering}
                  onChange={(e) => setInvertSteering(e.target.checked)}
                  color="secondary"
                  size="small"
                />
              }
              label={<Typography style={{ color: Colors.white, marginRight: "15px" }}>Invert</Typography>}
            />
          )}
          <FormControlLabel
            control={
              <Switch
                checked={controllerEnabled}
                onChange={(e) => setControllerEnabled(e.target.checked)}
                color="primary"
              />
            }
            label={<Typography style={{ color: Colors.white }}>Enable</Typography>}
          />
        </div>
      </div>

      <Typography style={{ color: controllerConnected ? Colors.green500 : Colors.red500, marginBottom: "10px" }}>
        Status: {controllerStatus} {controllerConnected && invertSteering && !useVirtualControls && <span style={{ color: Colors.orange500 }}>(Steering Inverted)</span>}
      </Typography>

      {controllerConnected && controllerEnabled && !useVirtualControls && (
        <div>
          <div style={{ display: "flex", marginBottom: "5px" }}>
            <Typography style={{ color: Colors.white, width: "80px" }}>Steering:</Typography>
            <div
              style={{
                flex: 1,
                height: "20px",
                backgroundColor: Colors.grey900,
                position: "relative",
                borderRadius: "4px",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  position: "absolute",
                  left: "50%",
                  top: 0,
                  bottom: 0,
                  width: "2px",
                  backgroundColor: Colors.white30,
                }}
              />
              <div
                style={{
                  position: "absolute",
                  left: `${50 + steering * 50}%`,
                  top: 0,
                  bottom: 0,
                  width: "10px",
                  backgroundColor: Math.abs(steering) > 0.7 ? Colors.red500 : Colors.blue500,
                  transform: "translateX(-50%)",
                  transition: "left 0.1s ease-out",
                }}
              />
            </div>
          </div>

          <div style={{ display: "flex" }}>
            <Typography style={{ color: Colors.white, width: "80px" }}>Throttle:</Typography>
            <div
              style={{
                flex: 1,
                height: "20px",
                backgroundColor: Colors.grey900,
                position: "relative",
                borderRadius: "4px",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  position: "absolute",
                  left: 0,
                  top: 0,
                  bottom: 0,
                  width: `${Math.max(0, throttle) * 100}%`,
                  backgroundColor: throttle > 0.7 ? Colors.red500 : Colors.green500,
                  transition: "width 0.1s ease-out",
                }}
              />
              <div
                style={{
                  position: "absolute",
                  right: 0,
                  top: 0,
                  bottom: 0,
                  width: `${Math.abs(Math.min(0, throttle)) * 100}%`,
                  backgroundColor: Colors.orange500,
                  transition: "width 0.1s ease-out",
                }}
              />
            </div>
          </div>

          {isMobile && (
            <Typography
              style={{
                color: Colors.yellow500,
                backgroundColor: Colors.grey900,
                padding: "8px",
                borderRadius: "5px",
                fontSize: "0.9rem",
                marginTop: "10px",
                display: "flex",
                alignItems: "center",
              }}
            >
              <span style={{ marginRight: "5px" }}>⚠️</span>
              <span>If your controller causes navigation issues, try tapping the screen once first.</span>
            </Typography>
          )}
        </div>
      )}

      {controllerEnabled && useVirtualControls && renderVirtualController()}

      {(!controllerConnected && !useVirtualControls) && (
        <div>
          <Typography style={{ color: Colors.white60, marginBottom: "10px" }}>
            Connect an Xbox controller to use this feature
          </Typography>
          <Typography
            style={{
              color: Colors.yellow500,
              backgroundColor: Colors.grey900,
              padding: "8px",
              borderRadius: "5px",
              fontSize: "0.9rem",
              display: "flex",
              alignItems: "center",
            }}
          >
            <span style={{ marginRight: "5px" }}>⚠️</span>
            <span>Important: Press any button on your controller after connecting to activate it</span>
          </Typography>
          {isMobile && (
            <Typography
              style={{
                color: Colors.yellow500,
                backgroundColor: Colors.grey900,
                padding: "8px",
                borderRadius: "5px",
                fontSize: "0.9rem",
                marginTop: "10px",
                display: "flex",
                alignItems: "center",
              }}
            >
              <span style={{ marginRight: "5px" }}>ℹ️</span>
              <span>Make sure your controller is paired with your device before enabling.</span>
            </Typography>
          )}
          <Button
            variant="contained"
            color="secondary"
            fullWidth
            style={{ marginTop: "15px" }}
            onClick={() => safeSetUseVirtualControls(true)}
          >
            Use Virtual Controller Instead
          </Button>
        </div>
      )}

      {(!controllerConnected && useVirtualControls && !controllerEnabled) && (
        <Typography style={{ color: Colors.white60 }}>
          Enable controller to start using virtual controls
        </Typography>
      )}
    </Paper>
  );
}

export default Joystick;
