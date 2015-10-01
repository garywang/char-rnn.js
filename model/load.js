"use strict";

const Promise = require("promise");

const Memory = require("./memory");
const Linalg = require("./linalg");
const Lstm = require("./lstm");

function loadFromUrl(path) {
  function get(url, type) {
    return new Promise(function(resolve, reject) {
      var req = new XMLHttpRequest();
      req.open("GET", url, true);
      req.responseType = type;
      req.onload = function() {
        resolve(req.response);
      };
      req.onerror = function() {
        reject();
      };
      req.send();
      return req;
    });
  }

  return Promise.all([get(path + ".dat", "arraybuffer"),
                      get(path + ".json", "json")])
    .then(function(res) {
      console.log("Got data " + path);
      return load(res[0], res[1]);
    });
}

function load(buffer, metadata) {
  var memory = new Memory(buffer, metadata.next);
  var linalg = new Linalg(memory);

  var arrays = []
  for (var n = 0; n < metadata.arrays.length; n++) {
    var array =
      new Float32Array(buffer, metadata.arrays[n].offset,
                       metadata.arrays[n].length);
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
    affines.push(
      linalg.makeAffineTransformation(arrays[2 * n], arrays[2 * n + 1]));
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

  var model = new Lstm(linalg, params);
  return model;
}

exports.loadFromUrl = loadFromUrl;
exports.load = load;
