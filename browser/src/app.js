import { adFrameUrl } from "./site-config.js";
import { MAX_CHARS, MIN_WORDS, countWords } from "./text.js";

const byId = (id) => document.getElementById(id);
const form = byId("editor");
const manuscript = byId("manuscript");
const fileInput = byId("file");
const fileButton = byId("file-button");
const example = byId("example");
const run = byId("run");
const clear = byId("clear");
const status = byId("status");
const number = new Intl.NumberFormat("ru-RU");
const plurals = new Intl.PluralRules("ru-RU");
const scoreNumber = new Intl.NumberFormat("ru-RU", { maximumFractionDigits: 1 });
let worker;
let busy = false;
let inputVersion = 0;

const sample = `К полуночи трамваи перестали ходить, и с улицы стало слышно, как внизу, в кухне, капает из крана. Лида сняла пальто, но вешать не стала. В кармане ещё лежали два билета. Она вынула один, разгладила на столе и подложила под сахарницу, чтобы не свернулся обратно.
— На какой ряд? — спросил отец из комнаты.
— На пятый.
— Хороший ряд. Оттуда и суфлёра не слышно.
Лида постояла у двери. Сказать, что они так и не вошли в зал, можно было бы и утром. Отец всё равно уже искал очки: ему хотелось увидеть, какое нынче время печатают на билетах. Она сняла второй билет с ладони и пошла к нему.`;

function updateCounter() {
  const text = manuscript.value;
  const words = countWords(text);
  const wordForm = { one: "слово", few: "слова", many: "слов" }[plurals.select(words)];
  const charForm = { one: "знак", few: "знака", many: "знаков" }[plurals.select(text.length)];
  byId("counter").textContent = `${number.format(words)} ${wordForm} · ${number.format(text.length)} ${charForm}`;
  clear.disabled = busy || !text;
}

function showStatus(message = "", isError = false) {
  status.textContent = message;
  status.hidden = !message;
  status.toggleAttribute("data-error", isError);
}

function resetReport() {
  byId("report-empty").hidden = false;
  byId("report-data").hidden = true;
  byId("fragment-map").replaceChildren();
  byId("fragments").replaceChildren();
}

function inputChanged() {
  inputVersion += 1;
  updateCounter();
  showStatus();
  resetReport();
}

function setBusy(value) {
  busy = value;
  manuscript.readOnly = value;
  for (const control of [run, example, fileButton]) control.disabled = value;
  run.setAttribute("aria-busy", String(value));
  updateCounter();
}

function renderReport(results) {
  const map = byId("fragment-map");
  const fragments = byId("fragments");
  const assessed = results.filter(({ short }) => !short);
  const count = assessed.filter(({ high }) => high).length;
  if (count) {
    byId("result-head").textContent = `Выше порога модели: ${count} из ${assessed.length}`;
  } else {
    byId("result-head").textContent = assessed.length
      ? "Сильных сигналов модель не нашла"
      : "Фрагменты слишком короткие для оценки";
  }
  byId("result-caption").textContent = "Баллы по фрагментам от 0 до 100. Высокий балл — повод перечитать отрывок. Модель может ошибаться.";
  results.forEach(({ text, score, high, short }, index) => {
    const n = index + 1;
    const label = `Фрагмент ${n}`;
    const scoreText = short ? "Короткий отрывок" : scoreNumber.format(score * 100) + " / 100";
    const button = document.createElement("button");
    const fragment = document.createElement("details");
    const heading = document.createElement("summary");
    const value = document.createElement("span");
    const body = document.createElement("p");

    button.type = "button";
    button.textContent = n;
    button.setAttribute("aria-label", `${label}, ${scoreText}`);
    if (high) {
      button.dataset.signal = "high";
      fragment.dataset.signal = "high";
    }
    fragment.className = "fragment";
    heading.append(label);
    value.className = "fragment-value";
    value.textContent = scoreText;
    heading.append(value);
    body.className = "fragment-body";
    body.textContent = text;
    fragment.append(heading, body);

    button.addEventListener("click", () => {
      fragment.open = true;
      heading.focus({ preventScroll: true });
      fragment.scrollIntoView({ block: "nearest" });
    });
    map.append(button);
    fragments.append(fragment);
  });
  byId("report-empty").hidden = true;
  byId("report-data").hidden = false;
  byId("report-title").focus({ preventScroll: true });
  byId("report").scrollIntoView({ block: "nearest" });
}

function modelError() {
  worker?.terminate();
  worker = undefined;
  setBusy(false);
  showStatus("Не удалось запустить модель. Проверьте соединение и попробуйте снова.", true);
}

function getWorker() {
  if (!worker) {
    worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    worker.addEventListener("error", modelError);
    worker.addEventListener("message", ({ data }) => {
      switch (data.kind) {
        case "loading":
          showStatus("Загружаем модель в браузер. При первом запуске это может занять минуту.");
          break;
        case "progress":
          showStatus(`Проверяем фрагмент ${data.current} из ${data.total}…`);
          break;
        case "done":
          setBusy(false);
          showStatus();
          renderReport(data.results);
          break;
        case "error":
          modelError();
          break;
      }
    });
  }
  return worker;
}

manuscript.addEventListener("input", inputChanged);
example.addEventListener("click", () => {
  if (manuscript.value.trim()) {
    showStatus("Очистите поле, чтобы вставить пример.");
    return;
  }
  manuscript.value = sample;
  inputChanged();
  manuscript.focus();
  manuscript.setSelectionRange(0, 0);
  manuscript.scrollTop = 0;
});
clear.addEventListener("click", () => {
  manuscript.value = "";
  inputChanged();
  manuscript.focus();
});
fileButton.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", async () => {
  const [file] = fileInput.files;
  fileInput.value = "";
  if (!file) return;
  const version = ++inputVersion;
  try {
    if (!/\.txt$/iu.test(file.name) || file.size > MAX_CHARS * 4 + 3) {
      throw new Error("size");
    }
    const bytes = await file.arrayBuffer();
    const view = new Uint8Array(bytes);
    const isUtf16 = (view[0] === 0xff && view[1] === 0xfe) ||
      (view[0] === 0xfe && view[1] === 0xff);
    let text;
    try {
      const encoding = isUtf16 ? (view[0] === 0xff ? "utf-16le" : "utf-16be") : "utf-8";
      text = new TextDecoder(encoding, { fatal: true }).decode(bytes);
    } catch {
      if (isUtf16) throw new Error("encoding");
      text = new TextDecoder("windows-1251").decode(bytes);
    }
    if (busy || version !== inputVersion) return;
    if (text.length > MAX_CHARS) throw new Error("size");
    if (!text.trim() || /\u0000/u.test(text)) throw new Error("encoding");
    manuscript.value = text;
    inputChanged();
    manuscript.focus();
  } catch (error) {
    if (busy || version !== inputVersion) return;
    showStatus(error.message === "size"
      ? "Выберите TXT-файл до 40 000 знаков."
      : "Не удалось прочитать текст. Проверьте, что TXT-файл содержит текст.", true);
  }
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (busy) return;
  const text = manuscript.value.trim();
  const letters = text.match(/\p{L}/gu)?.length ?? 0;
  const russian = text.match(/[а-яё]/giu)?.length ?? 0;
  if (manuscript.value.length > MAX_CHARS) {
    showStatus("Сократите текст до 40 000 знаков.", true);
  } else if (countWords(text) < MIN_WORDS) {
    showStatus("Добавьте больше текста. Для проверки нужно от 80 слов.", true);
  } else if (russian < letters * 0.6) {
    showStatus("Для проверки нужен текст преимущественно на русском языке.", true);
  } else {
    resetReport();
    setBusy(true);
    showStatus("Подготавливаем проверку…");
    try {
      getWorker().postMessage(text);
    } catch {
      modelError();
    }
    return;
  }
  manuscript.focus();
});

document.querySelector(".header-link").addEventListener("click", () => {
  byId("method").open = true;
});

if (adFrameUrl) {
  try {
    const url = new URL(adFrameUrl);
    if (url.protocol === "https:" && url.origin !== location.origin) {
      const frame = byId("ad-slot");
      frame.src = url.href;
      frame.hidden = false;
    }
  } catch {
    // An invalid advertising URL leaves the placement hidden.
  }
}
updateCounter();
