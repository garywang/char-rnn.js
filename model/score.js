var utils = require("./utils");

module.exports = function(model) {
  "use strict"

  return function score(str, initialState) {
    var bytes = utils.stringToBytes(str);
    var state = initialState ?
      model.copyState(initialState) :
      model.makeState();

    var currentScore = 0;
    for (var n = 0; n < bytes.length; n++) {
      if (n > 0) {
        var probs = model.predict(state);
        currentScore += Math.log(probs[model.byteToIndex(bytes[n])]);
      }
      model.forward(state, bytes[n], state);
    }

    return currentScore;
  }
}
