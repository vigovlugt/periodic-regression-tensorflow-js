const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));

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

const xs = tf.tensor1d([0,1,2,3,4,5,6,7,8,9]);
const ys = tf.tensor1d([0,1,2,1,0,1,2,1,0,1]);

train(xs,ys);
// ys.print();
// predict(xs).print();
console.log("-------")
a.print()
b.print()
c.print()
d.print();
