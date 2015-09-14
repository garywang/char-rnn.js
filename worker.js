var assert = require("./assert");

function stringToBytes(str) {
  return unescape(encodeURIComponent(str));
}

function bytesToString(bytes) {
  return decodeURIComponent(escape(bytes));
}

function load(path, callback) {
  function get(url, type, callback) {
    var req = new XMLHttpRequest();
    req.open("GET", url, true);
    req.responseType = type;
    req.onload = function() {
      callback(req.response);
    }
    req.send();
    return req;
  }

  get(path + ".dat?", "arraybuffer", function(buffer) {
    get(path + ".json", "json", function(metadata) {
      console.log("Got data");
      init(buffer, metadata);
      if (callback) {
        callback();
      }
    });
  });
}

function init(buffer, metadata) {
  var memory = require("./memory")(buffer, metadata.next);
  var linalg = require("./linalg")(memory);

  var arrays = []
  for (var n = 0; n < metadata.arrays.length; n++) {
    var array =
      new Float32Array(buffer, metadata.arrays[n].offset, metadata.arrays[n].length);
    var dims = metadata.arrays[n].dim;
    if (dims.length > 1) {
      array.nRows = dims[0];
      array.nCols = dims[1];
    }
    array.dims = dims;
    arrays.push(array);
  }

  var affines = [];
  for (var n = 0; n < arrays.length / 2; n++) {
    affines.push(linalg.makeAffineTransformation(arrays[2 * n], arrays[2 * n + 1]));
  }

  var params = {};
  params.affines = affines;
  params.nNodes = affines[0].outLength / 4;
  params.nLayers = (affines.length - 1) / 2;

  params.vocab = {};
  params.ivocab = {};
  for (var n = 0; n < metadata.vocab.length; n++) {
    if (metadata.vocab[n] !== null) {
      params.vocab[String.fromCharCode(n)] = metadata.vocab[n];
      params.ivocab[metadata.vocab[n]] = String.fromCharCode(n);
    }
  }

  var model = require("./lstm")(memory, linalg, params);

  var d = new Date();
  for (var n = 0; n < 100; n++) {
    model.predict(model.forward(model.makeState(), "a", model.makeState()));
  }
  console.log(new Date() - d);
  console.log(model.predict(model.forward(model.makeState(), "a", model.makeState())));
  console.log(model.predict(model.forward(model.makeState(), "a", model.makeState())));

  global.score = require("./score")(memory, model);

  var d = new Date();
  for (var n = 0; n < 10; n++) {
    global.score(" this is some text foo bar baz");
  }
  console.log(new Date() - d);

  return { memory: memory, model: model };
}

//load("data/large");

global.load = load;
