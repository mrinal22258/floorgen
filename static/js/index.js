$(document).ready(function() {
  $('#copy-bibtex-btn').click(function() {
    var code = $('#bibtex-code').text();
    navigator.clipboard.writeText(code).then(function() {
      var btn = $('#copy-bibtex-btn');
      btn.html('<span class="icon"><i class="fas fa-check"></i></span><span>Copied!</span>');
      setTimeout(function() {
        btn.html('<span class="icon"><i class="fas fa-copy"></i></span><span>Copy BibTeX</span>');
      }, 2000);
    });
  });
});
