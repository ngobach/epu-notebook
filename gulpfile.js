const gulp = require('gulp');
const del = require('del');
const fs = require('fs');
const hightlight = require('highlight.js');

const es = require('event-stream');
const util = require('gulp-util');
const markdown = require('gulp-markdown');
const clone = require('gulp-clone');
const rename = require('gulp-rename');
const transform = require('gulp-transform');
/* Tasks */

gulp.task('default', ['build']);

gulp.task('clean', function() {
  return del('./dist');
});

gulp.task('build', ['clean'], function() {
  const template = fs.readFileSync('./src/template.html').toString('utf8');
  const html = gulp.src('./src/epu-notebook.md')
    .pipe(markdown({
      highlight: function(code, lang) {
        return hightlight.highlight(lang || 'cpp', code).value;
      }
    }))
    .pipe(transform('utf8', function (content, file) {
      return template
        .replace('{BODY}', content)
        .replace('{TIME}', (new Date()).toLocaleString('vi-VN'));
    }));
  const cloned = html
    .pipe(clone())
    .pipe(rename('index.html'));
  const css = gulp.src('./src/style.css');
  es.merge(html, cloned, css)
    .pipe(gulp.dest('./dist'));
});
