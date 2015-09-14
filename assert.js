module.exports = function assert(condition) {
  if (!condition) {
    throw new Error("assertion failed");
  }
}
