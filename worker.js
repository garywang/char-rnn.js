var assert = require("assert");
var Promise = require("promise");

var charRnn = require("./char-rnn");
var Checker = require("./checker");

function load(smallUrl, largeUrl) {
  return Promise.all([charRnn.loadFromUrl(smallUrl),
                      charRnn.loadFromUrl(largeUrl)])
    .then(function(models) {
      global.model = models[1];
      global.score = models[1].score.bind(models[1]);

      console.log("foo");
      var d = new Date();
      for (var n = 0; n < 100; n++) {
        model.sample(model.getState("a"));
      }
      console.log(new Date() - d);
      console.log(model.sample(model.getState("appl")));
      console.log(model.sample(model.getState("appl")));
      console.log(model.sample(model.getState("appl")));
      console.log(model.score("apple"));

      var checker = new Checker(models[0], models[1]);
      global.checker = checker;
      global.check = checker.check.bind(checker);

      return model;
    });
}

//load("data/large");

global.load = load;
global.charRnn = charRnn;
