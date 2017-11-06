const gulp = require('gulp');
const del = require('del');
const fs = require('fs');

const es = require('event-stream');
const util = require('gulp-util');
const markdown = require('gulp-markdown');
const clone = require('gulp-clone');
const md2pdf = require('gulp-markdown-pdf');
const rename = require('gulp-rename');
/* Tasks */

gulp.task('default', ['build']);

gulp.task('clean', function() {
  return del('./dist');
});

gulp.task('build', ['clean'], function() {
  const renderer = new markdown.marked.Renderer();
  const style = fs.readFileSync('./src/style.css');
  renderer.html = function(html) {
    return '<style>' + style + '</style>' + html;
  }
  const src = gulp.src('./src/epu-notebook.md')
    .pipe(markdown({
      renderer: renderer
    }));
  const cloned = src
    .pipe(clone())
    .pipe(rename('index.html'));
  es.merge(src, cloned)
    .pipe(gulp.dest('./dist'));
});
