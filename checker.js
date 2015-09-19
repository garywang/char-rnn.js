var assert = require("assert");

module.exports = function Checker(smallModel, largeModel) {
  "use strict";

  function compare(contextBefore, contextAfter, original, replacement) {
    function compareWithModel(model) {
      model.memory.pushFrame();
      var state = model.getState(contextBefore);
      var diff = model.score(replacement + contextAfter, state) -
        model.score(original + contextAfter, state);
      var priorDiff = model.score(replacement) - model.score(original);
      if (priorDiff > 0) {
        diff -= priorDiff;
      }
      model.memory.popFrame();
      return diff;
    }
    var smallDiff = compareWithModel(smallModel);
    if (smallDiff < -1.) {
      return smallDiff;
    }
    console.log("large");
    return compareWithModel(largeModel);
  }

  const REPLACEMENTS = (function() {
    var sets = [
      ["its", "it's"],
      ["his", "he's"],
      ["there", "their", "they're"],
    ];

    function capitalize(s) {
      return s[0].toUpperCase() + s.slice(1);
    }

    var replacements = {};
    for (var i = 0; i < sets.length; i++) {
      for (var j = 0; j < sets[i].length; j++) {
        replacements[sets[i][j]] = [];
        replacements[capitalize(sets[i][j])] = [];
        for (var k = 0; k < sets[i].length; k++) {
          if (j != k) {
            replacements[sets[i][j]].push(sets[i][k]);
            replacements[capitalize(sets[i][j])].push(capitalize(sets[i][k]));
          }
        }
      }
    }
    return replacements;
  })();

  const PATTERN = (function() {
    var delimiters = "[ ,.?!'\"/();\n]";
    return new RegExp(Object.keys(REPLACEMENTS)
                      .map(s => delimiters + s + delimiters)
                      .join("|"), "g");
  })();

  console.log(REPLACEMENTS);
  console.log(PATTERN);

  const CONTEXT_LENGTH = 50;

  function check(str) {
    str = "\n" + str + "\n";

    var match;
    while ((match = PATTERN.exec(str)) !== null) {
      var contextBefore = str.slice(Math.max(0, match.index - CONTEXT_LENGTH),
                                    match.index);
      var original = match[0];
      var contextAfter = str.substr(match.index + original.length,
                                    CONTEXT_LENGTH);

      var key = original.slice(1, -1);
      console.log(key);
      assert(REPLACEMENTS[key]);

      var bestScore = 1.;
      var best = null;
      for (var i = 0; i < REPLACEMENTS[key].length; i++) {
        if (REPLACEMENTS[key][i] != key) {
          var replacement =
            original[0] + REPLACEMENTS[key][i] + original[original.length - 1];
          console.log(replacement);
          var score = compare(contextBefore, contextAfter, original, replacement);
          console.log(score);
          if (score > bestScore) {
            bestScore = score;
            best = replacement;
          }
        }
      }
      if (best !== null) {
        str = str.slice(0, match.index) + best +
          str.slice(match.index + original.length);
      }
    }

    return str.slice(1, -1);
  }

  return { check: check };
}
