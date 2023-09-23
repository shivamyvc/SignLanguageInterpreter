const video3 = document.getElementsByClassName('input_video3')[0];
const out3 = document.getElementsByClassName('output3')[0];
const controlsElement3 = document.getElementsByClassName('control3')[0];
const canvasCtx3 = out3.getContext('2d');
const fpsControl = new FPS();

const spinner = document.querySelector('.loading');
spinner.ontransitionend = () => {
  spinner.style.display = 'none';
};

function onResultsHands(results) {
  document.body.classList.add('loaded');
  fpsControl.tick();
  canvasCtx3.save();
  let w=out3.width;
  let h= out3.height;
  if (results.multiHandLandmarks && results.multiHandedness) {
    let coordinates=[];
    for (let index = 0; index < results.multiHandLandmarks.length; index++) {
      const classification = results.multiHandedness[index];
      const isRightHand = classification.label === 'Right';
      const landmarks = results.multiHandLandmarks[index];
      drawConnectors(
          canvasCtx3, landmarks, HAND_CONNECTIONS,
          {color: isRightHand ? '#FFFFFF' : '#FFFFFF'}),
      drawLandmarks(canvasCtx3, landmarks, {
        color: isRightHand ? '#FFFFFF' : '#FFFFFF',
        fillColor: isRightHand ? '#FFFFFF' : '#FFFFFF',
        radius:1
      });
      
      if (results.multiHandLandmarks.length==2){
        coordinates.push(find_coord(landmarks,w,h));
        
      }
      else{
        let one_hand=find_coord(landmarks,w,h);
        // canvasCtx3.drawImage(
        //   results.image, 0, 0, out3.width, out3.height);
        canvasCtx3.strokeStyle = "red";
        canvasCtx3.lineWidth = 3;
        canvasCtx3.strokeRect(one_hand[0],one_hand[1],one_hand[2],one_hand[3]);
        canvasCtx3.clearRect(0, 0, out3.width, out3.height);

        sendFrame(out3);
      }
    }
    if (coordinates.length!=0){
      let x_min=Math.min(coordinates[0][0],coordinates[1][0])
      let y_min=Math.min(coordinates[0][1],coordinates[1][1])
      let x_max=Math.max(coordinates[0][2],coordinates[1][2])
      let y_max=Math.max(coordinates[0][3],coordinates[1][3])
      // canvasCtx3.drawImage(
      //   results.image, 0, 0, out3.width, out3.height);
      canvasCtx3.strokeStyle = "red";
      canvasCtx3.lineWidth = 3;
      canvasCtx3.strokeRect(x_min,y_min,x_max,y_max);
      
      sendFrame(out3);
    }
  }
  canvasCtx3.restore();
}

function find_coord(hand,w,h){
    let x_max = 0;
    let y_max = 0;
    let x_min = w;
    let y_min = h;
    for(let lm=0; lm < hand.length; lm++){
      let x= parseInt((hand[lm].x)* (w * 1.09));
      let y=parseInt((hand[lm].y)* (h* 1.09));
      if (x < w && x > x_max ){
        x_max = parseInt(x);}
      if (x < x_min && x > 0){
      x_min = parseInt(x-(x*0.43));}
      if (y > y_max && y < h){
      y_max = parseInt(y);}
      if (y < y_min && y>0){
      y_min = parseInt(y-(y*0.48));
      }

    }
    return [x_min,y_min,x_max,y_max];
          
}


function sendFrame(out3) {

  const imageData = out3.toDataURL("image/jpeg");
  

  
  // Send the image data to the Django backend using AJAX
  $.ajax({
      type: "POST",
      url: "{% url '/get_frames/' %}",
      data: {
          csrfmiddlewaretoken: "{{ csrf_token }}",
          image_data: imageData,
         
      },
      success: function(response) {
          // Update the processed image element with the received image data
          // processedImage.src = "data:image/jpeg;base64," + response.processed_image_data;
          // processedImage.style.display = "inline";
          console.log(success)
      },
      error: function(xhr, status, error) {
          console.error("Error sending frame to server:", error);
      }
  });
}




const hands = new Hands({locateFile: (file) => {
  return `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.1/${file}`;
}});
hands.onResults(onResultsHands);

const camera = new Camera(video3, {
  onFrame: async () => {
    await hands.send({image: video3});
  },
  width: 680,
  height: 480
  
});
camera.start();

new ControlPanel(controlsElement3, {
      selfieMode: true,
      maxNumHands: 2,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    })
    .add([
      new StaticText({title: 'MediaPipe Hands'}),
      fpsControl,
      new Toggle({title: 'Selfie Mode', field: 'selfieMode'}),

    ])
    .on(options => {
      video3.classList.toggle('selfie', options.selfieMode);
      hands.setOptions(options);
    });