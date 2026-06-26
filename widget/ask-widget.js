// Embeddable ask box for docs.fabrix.ai: POSTs to FastAPI /ask, renders answer + source links.
(function () {
  "use strict";

  var script = document.currentScript;
  var globalConfig = window.FabrixAskConfig || {};

  function apiUrl() {
    // data-api-url on script tag, or FabrixAskConfig, or localhost for dev
    if (script && script.getAttribute("data-api-url")) {
      return script.getAttribute("data-api-url").replace(/\/$/, "");
    }
    if (globalConfig.apiUrl) {
      return globalConfig.apiUrl.replace(/\/$/, "");
    }
    return "http://localhost:8080";
  }

  function apiKey() {
    return globalConfig.apiKey || null;
  }

  function mountParent() {
    var containerId = script && script.getAttribute("data-container");
    if (containerId) {
      var el = document.getElementById(containerId);
      if (el) {
        return el;
      }
    }
    return document.body;
  }

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text !== undefined) {
      node.textContent = text;
    }
    return node;
  }

  function setVisible(node, visible) {
    if (visible) {
      node.removeAttribute("hidden");
    } else {
      node.setAttribute("hidden", "");
    }
  }

  function renderAnswer(container, answer) {
    container.innerHTML = "";
    var parts = (answer || "").split(/\n\n+/);
    parts.forEach(function (part) {
      var trimmed = part.trim();
      if (trimmed) {
        container.appendChild(el("p", null, trimmed));
      }
    });
  }

  function renderSources(list, sources) {
    list.innerHTML = "";
    (sources || []).forEach(function (source) {
      var li = el("li");
      var link = document.createElement("a");
      link.href = source.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = source.title || source.url;
      li.appendChild(link);
      list.appendChild(li);
    });
  }

  var root = el("div", "fabrix-ask");
  root.id = "fabrix-ask-widget";

  var form = el("form", "fabrix-ask-form");
  var input = el("input", "fabrix-ask-input");
  input.type = "text";
  input.placeholder = "Ask a question about the documentation...";
  input.setAttribute("aria-label", "Question");

  var button = el("button", "fabrix-ask-button", "Ask");
  button.type = "submit";

  form.appendChild(input);
  form.appendChild(button);

  var loading = el("div", "fabrix-ask-loading", "Loading...");
  loading.setAttribute("hidden", "");

  var errorBox = el("div", "fabrix-ask-error");
  errorBox.setAttribute("hidden", "");

  var answerBox = el("div", "fabrix-ask-answer");
  var sourcesList = el("ul", "fabrix-ask-sources");

  root.appendChild(form);
  root.appendChild(loading);
  root.appendChild(errorBox);
  root.appendChild(answerBox);
  root.appendChild(sourcesList);
  mountParent().appendChild(root);

  form.addEventListener("submit", function (event) {
    // POST {question} to /ask, show answer + docs.fabrix.ai source links
    event.preventDefault();
    var question = input.value.trim();
    if (!question) {
      return;
    }

    setVisible(errorBox, false);
    setVisible(loading, true);
    answerBox.innerHTML = "";
    sourcesList.innerHTML = "";
    input.disabled = true;
    button.disabled = true;

    var headers = { "Content-Type": "application/json" };
    var key = apiKey();
    if (key) {
      headers["X-API-Key"] = key;
    }

    fetch(apiUrl() + "/ask", {
      method: "POST",
      headers: headers,
      body: JSON.stringify({ question: question }),
    })
      .then(function (response) {
        if (!response.ok) {
          return response.text().then(function (body) {
            throw new Error("Request failed (" + response.status + "): " + body);
          });
        }
        return response.json();
      })
      .then(function (data) {
        renderAnswer(answerBox, data.answer);
        renderSources(sourcesList, data.sources);
      })
      .catch(function (err) {
        errorBox.textContent = err.message || "Request failed.";
        setVisible(errorBox, true);
      })
      .finally(function () {
        setVisible(loading, false);
        input.disabled = false;
        button.disabled = false;
      });
  });
})();
