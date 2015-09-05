
function assert(condition) {
  if (!condition) {
    throw new Error("assertion failed");
  }
}

window.addEventListener("load", function() {
  //var worker = new Worker("worker.js");
  load("data/large");
});
