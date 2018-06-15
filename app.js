const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));

let dataArr = JSON.parse(data).data;

let xs = null;
let ys = null;

function predict(x) {
  // y = a * sin(b(x-c)) + d
  return tf.tidy(() => {
    return a.mul(tf.sin(b.mul(tf.sub(x,c)))).add(d);
  });
}

function loss(predictions, labels) {
  const meanSquareError = predictions.sub(labels).square().mean();
  return meanSquareError;
}

function train(xs, ys, numIterations = 1000) {

  const learningRate = 0.5;
  const optimizer = tf.train.sgd(learningRate);

  for (let iter = 0; iter < numIterations; iter++) {
    optimizer.minimize(() => {
      const predsYs = predict(xs);
      return loss(predsYs, ys);
    });
    if(iter % 100 == 0)
      console.log(iter);
  }
}

let xArr = [];
let xValues = [];
let yArr = [];

function getTrainData(){
  for (let i = 0; i < dataArr.length; i++) {
    let date = dataArr[i]["Datum"].split("-");
    let time = dataArr[i]["Tijd"].split(":");
    let dateObj = new Date(date[2],date[1],date[0],time[0],time[1]);
    xArr.push((dateObj.getTime() / 1000 - 1530873600) / 60)

    if(isNaN(dataArr[i]["Meting"]) || dataArr[i]["Meting"] == null ){
      xArr.shift();
    }
    else{
      yArr.push(dataArr[i]["Meting"]);
    }
  }
  xValues = xArr;
  xs = tf.tensor1d(xArr);
  ys = tf.tensor1d(yArr); 
}

getTrainData();
train(xs,ys);

let chart;

window.onload = ()=>{
  let chartElement = document.getElementById("chart").getContext("2d");

  let predictionData = predict(tf.tensor1d(xValues)).dataSync();

  let actualData = yArr;
  chart = new Chart(chartElement,{
    type:"line",
    data:{
      labels: xValues,
      datasets:[
        {
          label:"Prediction",
          data: predictionData,
          backgroundColor:"rgba(255,0,0,.5)"
        },
        {
          label:"Actual",
          data: actualData,
          backgroundColor:"rgba(0,255,0,.5)"
        }
      ]
    }
  });
}

console.log("-------")
a.print()
b.print()
c.print()
d.print()
