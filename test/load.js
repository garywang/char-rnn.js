'use strict';

const chai = require('chai');
chai.use(require('chai-as-promised'));
const should = chai.should();
const fs = require('fs');
const Promise = require('bluebird');
const path = require('path');

const CharRnn = require('../lib/char-rnn.js');

const {load, loadFromUrl} = require('../lib/load');


describe('load', ()=> {
  it('takes a dat and json file with the same filename, and returns an instance of the CharRnn Model', ()=> {
    return Promise.all(
      [
        fs.readFileSync(path.join(__dirname, '../data/small.dat'), 'utf8'),
        fs.readFileSync(path.join(__dirname, '../data/small.json'), 'utf8')
      ])
      .spread( (dat, json) => [
          Buffer.from(dat),
          JSON.parse(json)
        ])
      .spread( (buffer, metaData) => load(buffer, metaData))
      .should.eventually.be.instanceof(CharRnn);
  })
})
