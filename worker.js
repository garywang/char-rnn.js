var assert = require("assert");
var Promise = require("promise");

var Model = require("./model");
var Checker = require("./checker");

function load(smallUrl, largeUrl) {
  return Promise.all([Model.loadFromUrl(smallUrl),
                      Model.loadFromUrl(largeUrl)])
    .then(function(models) {
      global.model = models[1];
      global.score = models[1].score;

      /*
      var d = new Date();
      for (var n = 0; n < 100; n++) {
        model.predict(model.forward(model.makeState(), "a", model.makeState()));
      }
      console.log(new Date() - d);
      console.log(model.predict(model.forward(model.makeState(), "a", model.makeState())));
      console.log(model.predict(model.forward(model.makeState(), "a", model.makeState())));
      */

      var checker = new Checker(models[0], models[1]);
      global.checker = checker;
      global.check = checker.check;

      return model;
    });
}

//load("data/large");

global.load = load;
