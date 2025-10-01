// script.js — TF.js model, training, prediction

let model = null;
let isTraining = false;

window.onload = async () => {
  try {
    updateStatus('Loading MovieLens data…');
    await loadData();
    populateUserDropdown();
    populateMovieDropdown();
    updateStatus('Data loaded. Training model…');
    await trainModel();
    updateStatus('✅ Model trained! Pick a user and a movie, then click Predict.');
    document.getElementById('predict-btn').disabled = false;
  } catch (e) {
    console.error(e);
    updateStatus('Failed: ' + e.message, true);
  }
};

function populateUserDropdown() {
  const sel = document.getElementById('user-select');
  sel.innerHTML = '';
  userIdList.forEach(id => {
    const opt = document.createElement('option');
    opt.value = id;
    opt.textContent = `User ${id}`;
    sel.appendChild(opt);
  });
}

function populateMovieDropdown() {
  const sel = document.getElementById('movie-select');
  sel.innerHTML = '';
  movies.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = m.year ? `${m.title} (${m.year})` : m.title;
    sel.appendChild(opt);
  });
}

function createModel(nUsers, nMovies, latentDim = 10) {
  const userInput = tf.input({ shape: [1], dtype: 'int32', name: 'userInput' });
  const movieInput = tf.input({ shape: [1], dtype: 'int32', name: 'movieInput' });

  const userEmbedding = tf.layers.embedding({
    inputDim: nUsers, outputDim: latentDim, name: 'userEmbedding'
  }).apply(userInput);

  const movieEmbedding = tf.layers.embedding({
    inputDim: nMovies, outputDim: latentDim, name: 'movieEmbedding'
  }).apply(movieInput);

  const userVector = tf.layers.flatten().apply(userEmbedding);
  const movieVector = tf.layers.flatten().apply(movieEmbedding);

  const dotProduct = tf.layers.dot({ axes: 1 }).apply([userVector, movieVector]);
  const prediction = tf.layers.reshape({ targetShape: [1] }).apply(dotProduct);

  return tf.model({ inputs: [userInput, movieInput], outputs: prediction });
}

async function trainModel() {
  isTraining = true;
  document.getElementById('predict-btn').disabled = true;

  const uIdx = new Int32Array(ratings.length);
  const mIdx = new Int32Array(ratings.length);
  const y   = new Float32Array(ratings.length);

  for (let i = 0; i < ratings.length; i++) {
    const r = ratings[i];
    uIdx[i] = userIdToIndex.get(r.userId);
    mIdx[i] = movieIdToIndex.get(r.movieId);
    y[i] = r.rating;
  }

  const userTensor  = tf.tensor2d(uIdx, [uIdx.length, 1], 'int32');
  const movieTensor = tf.tensor2d(mIdx, [mIdx.length, 1], 'int32');
  const ratingTensor= tf.tensor2d(y,   [y.length, 1], 'float32');

  model = createModel(numUsers, numMovies, 16);
  model.compile({ optimizer: tf.train.adam(0.001), loss: 'meanSquaredError' });

  updateStatus('Training model (8 epochs)…');
  await model.fit([userTensor, movieTensor], ratingTensor, {
    epochs: 8,
    batchSize: 64,
    validationSplit: 0.1,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        updateStatus(`Epoch ${epoch+1}/8 — loss: ${logs.loss.toFixed(4)}`);
      }
    }
  });

  tf.dispose([userTensor, movieTensor, ratingTensor]);
  isTraining = false;
}

async function predictRating() {
  if (!model || isTraining) {
    updateResult('Model not ready yet. Please wait…', 'medium');
    return;
  }
  const u = parseInt(document.getElementById('user-select').value, 10);
  const m = parseInt(document.getElementById('movie-select').value, 10);
  if (Number.isNaN(u) || Number.isNaN(m)) {
    updateResult('Please select both a user and a movie.', 'medium');
    return;
  }

  const uIndex = userIdToIndex.get(u);
  const mIndex = movieIdToIndex.get(m);

  const userTensor  = tf.tensor2d([[uIndex]], [1,1], 'int32');
  const movieTensor = tf.tensor2d([[mIndex]], [1,1], 'int32');
  const pred = model.predict([userTensor, movieTensor]);
  const val = (await pred.data())[0];

  const clamped = Math.max(1, Math.min(5, val));
  const movie = movies.find(x => x.id === m);
  const title = movie ? (movie.year ? `${movie.title} (${movie.year})` : movie.title) : `Movie ${m}`;

  let cls = 'medium';
  if (clamped >= 4) cls = 'high';
  else if (clamped <= 2) cls = 'low';

  updateResult(`Predicted rating for User ${u} on “${title}”: <strong>${clamped.toFixed(2)}/5</strong>`, cls);

  tf.dispose([userTensor, movieTensor, pred]);
}

function updateStatus(msg, isError = false) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.style.borderLeftColor = isError ? '#e74c3c' : '#3498db';
  el.style.background = isError ? '#fdedec' : '#f7f9fc';
}

function updateResult(html, cls = '') {
  const el = document.getElementById('result');
  el.innerHTML = html;
  el.className = `result ${cls}`;
}
