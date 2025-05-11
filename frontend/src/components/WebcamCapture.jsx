import React, { useRef, useEffect, useState } from 'react';
import axios from 'axios';

const WebcamStream = ({ apiUrl, onStop }) => {
  const videoRef = useRef(null);
  const [annotatedFrame, setAnnotatedFrame] = useState(null);
  const [streaming, setStreaming] = useState(true);
  // Store stream in a ref so it persists
  const streamRef = useRef(null);

  useEffect(() => {
    let intervalId;

    const startWebcam = async () => {
      if (!streamRef.current) {
        streamRef.current = await navigator.mediaDevices.getUserMedia({ video: true });
      }
      videoRef.current.srcObject = streamRef.current;
      intervalId = setInterval(captureAndSendFrame, 200); // 5 fps
    };

    const captureAndSendFrame = async () => {
      if (!videoRef.current) return;
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      canvas.getContext('2d').drawImage(videoRef.current, 0, 0);

      canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('image', blob, 'frame.jpg');
        try {
          const response = await axios.post(apiUrl, formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
          });
          if (response.data.status === 'success' && response.data.result_frame) {
            setAnnotatedFrame('data:image/jpeg;base64,' + response.data.result_frame);
          }
        } catch (err) {
          // handle error
        }
      }, 'image/jpeg', 0.95);
    };

    if (streaming) startWebcam();

    return () => {
      setStreaming(false);
      if (intervalId) clearInterval(intervalId);
      // Do not stop the stream here, so it persists across navigation
      // To fully persist, move stream management to a React context
    };
  }, [streaming, apiUrl]);

  return (
    <div>
      <video ref={videoRef} autoPlay playsInline style={{ display: annotatedFrame ? 'none' : 'block' }} />
      {annotatedFrame && <img src={annotatedFrame} alt="Annotated" />}
      <button onClick={onStop}>Stop</button>
    </div>
  );
};

export default WebcamStream;