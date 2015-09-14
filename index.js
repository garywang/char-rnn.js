
function assert(condition) {
  if (!condition) {
    throw new Error("assertion failed");
  }
}

window.addEventListener("load", function() {
  //var worker = new Worker("bundle.js");
  load("data/large");
});
