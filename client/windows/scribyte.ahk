#Requires AutoHotkey v2.0
#SingleInstance Force

global SCRIBYTE_API_URL := "http://127.0.0.1:8000"
global SCRIBYTE_HOLD_KEY := "F8"
global SCRIBYTE_PASTE_SHORTCUT := "^v"
global SCRIBYTE_STATUS_DURATION_MS := 3000
global SCRIBYTE_LONG_STATUS_DURATION_MS := 5000
global SCRIBYTE_RECORDING_STATUS_DURATION_MS := 1800
global SCRIBYTE_STATUS_RETRY_INTERVAL_MS := 5000
global scribyteIsRecording := false
global scribyteBackendReady := false
global scribyteStatusPollActive := false

A_TrayMenu.Delete()
A_TrayMenu.Add("Check Scribyte Status", ManualCheckScribyteStatus)
A_TrayMenu.Add()
A_TrayMenu.Add("Exit", ExitScribyte)

Hotkey("*" . SCRIBYTE_HOLD_KEY, StartDictation)
Hotkey("*" . SCRIBYTE_HOLD_KEY . " Up", StopDictation)

ShowStatus("Scribyte hotkey loaded. Hold " . SCRIBYTE_HOLD_KEY . " to dictate.")
SetTimer(CheckScribyteStatus, -10)


StartDictation(*) {
    global scribyteIsRecording

    if scribyteIsRecording {
        return
    }

    try {
        response := ApiRequest("POST", "/start_recording")
    } catch as err {
        ShowStatus("Could not reach Scribyte API.`n" . err.Message, SCRIBYTE_LONG_STATUS_DURATION_MS)
        return
    }

    if response["status"] != 200 {
        ShowStatus("Start failed.`n" . GetErrorMessage(response["body"]), SCRIBYTE_LONG_STATUS_DURATION_MS)
        RefreshRecordingState()
        return
    }

    scribyteIsRecording := JsonGetBoolean(response["body"], "recording", true)
    sampleRate := JsonGetNumber(response["body"], "sample_rate", 0)
    message := "Recording..."
    if sampleRate {
        message .= "`n" . sampleRate . " Hz"
    }
    ShowStatus(message, SCRIBYTE_RECORDING_STATUS_DURATION_MS)
}


StopDictation(*) {
    global scribyteIsRecording

    if !scribyteIsRecording {
        return
    }

    try {
        response := ApiRequest("POST", "/stop_recording_and_transcribe")
    } catch as err {
        RefreshRecordingState()
        ShowStatus("Could not reach Scribyte API.`n" . err.Message, SCRIBYTE_LONG_STATUS_DURATION_MS)
        return
    }

    if response["status"] != 200 {
        RefreshRecordingState()
        ShowStatus("Transcription failed.`n" . GetErrorMessage(response["body"]), SCRIBYTE_LONG_STATUS_DURATION_MS)
        return
    }

    scribyteIsRecording := false

    text := JsonGetString(response["body"], "text", "")
    chunkCount := JsonGetNumber(response["body"], "chunk_count", 0)
    latencySeconds := JsonGetNumber(response["body"], "latency_seconds", 0)

    if text = "" {
        ShowStatus("Transcription was empty.")
        return
    }

    PasteTranscription(text)

    summary := "Pasted dictation"
    if chunkCount {
        summary .= "`nChunks: " . chunkCount
    }
    if latencySeconds {
        summary .= " | Latency: " . Format("{1:.2f}s", latencySeconds)
    }
    ShowStatus(summary)
}


ManualCheckScribyteStatus(*) {
    CheckScribyteStatus(true)
}


CheckScribyteStatus(alwaysShowReady := true) {
    global scribyteBackendReady
    global scribyteIsRecording

    try {
        response := ApiRequest("GET", "/status")
    } catch as err {
        HandleBackendUnavailable("Could not reach Scribyte API.`n" . err.Message)
        return
    }

    if response["status"] != 200 {
        HandleBackendUnavailable("Status check failed.`n" . GetErrorMessage(response["body"]))
        return
    }

    ready := JsonGetBoolean(response["body"], "ready", false)
    recording := JsonGetBoolean(response["body"], "recording", false)
    device := JsonGetString(response["body"], "device", "unknown device")
    startupError := JsonGetString(response["body"], "startup_error", "")

    if !ready {
        message := "Scribyte backend is not ready."
        if startupError != "" {
            message .= "`n" . startupError
        }
        HandleBackendUnavailable(message)
        return
    }

    wasReady := scribyteBackendReady
    scribyteBackendReady := true
    scribyteIsRecording := recording
    StopStatusPolling()

    message := "Scribyte ready on " . device . "."
    if recording {
        message .= "`nA recording is already active."
    }
    if alwaysShowReady || !wasReady {
        ShowStatus(message)
    }
}


ExitScribyte(*) {
    ExitApp()
}


ApiRequest(method, path, body := "") {
    url := SCRIBYTE_API_URL . path
    request := ComObject("WinHttp.WinHttpRequest.5.1")
    request.SetTimeouts(2000, 2000, 10000, 60000)
    request.Open(method, url, false)
    request.SetRequestHeader("Accept", "application/json")

    if body != "" {
        request.SetRequestHeader("Content-Type", "application/json")
        request.Send(body)
    } else {
        request.Send()
    }

    return Map("status", request.Status, "body", request.ResponseText)
}


RefreshRecordingState() {
    global scribyteIsRecording
    global scribyteBackendReady

    try {
        response := ApiRequest("GET", "/status")
    } catch {
        scribyteIsRecording := false
        scribyteBackendReady := false
        return
    }

    if response["status"] != 200 {
        scribyteIsRecording := false
        scribyteBackendReady := false
        return
    }

    scribyteIsRecording := JsonGetBoolean(response["body"], "recording", false)
    scribyteBackendReady := JsonGetBoolean(response["body"], "ready", false)
}


HandleBackendUnavailable(message) {
    global scribyteBackendReady
    global scribyteIsRecording
    global scribyteStatusPollActive

    shouldNotify := scribyteBackendReady || !scribyteStatusPollActive
    scribyteBackendReady := false
    scribyteIsRecording := false
    StartStatusPolling()

    if shouldNotify {
        ShowStatus(message, SCRIBYTE_LONG_STATUS_DURATION_MS)
    }
}


StartStatusPolling() {
    global scribyteStatusPollActive

    if scribyteStatusPollActive {
        return
    }

    scribyteStatusPollActive := true
    SetTimer(CheckScribyteStatus, SCRIBYTE_STATUS_RETRY_INTERVAL_MS)
}


StopStatusPolling() {
    global scribyteStatusPollActive

    if !scribyteStatusPollActive {
        return
    }

    scribyteStatusPollActive := false
    SetTimer(CheckScribyteStatus, 0)
}


PasteTranscription(text) {
    savedClipboard := ClipboardAll()
    A_Clipboard := text

    if !ClipWait(1) {
        A_Clipboard := savedClipboard
        ShowStatus("Clipboard update timed out.")
        return
    }

    Send(SCRIBYTE_PASTE_SHORTCUT)
    SetTimer(RestoreClipboard.Bind(savedClipboard), -250)
}


RestoreClipboard(savedClipboard) {
    A_Clipboard := savedClipboard
}


ShowStatus(message, durationMs := "") {
    if durationMs = "" {
        durationMs := SCRIBYTE_STATUS_DURATION_MS
    }

    ToolTip(message)
    SetTimer(ClearStatus, -durationMs)
}


ClearStatus() {
    ToolTip()
}


GetErrorMessage(json) {
    detail := JsonGetString(json, "detail", "")
    if detail != "" {
        return detail
    }

    return "Unexpected API response."
}


JsonGetString(json, key, defaultValue := "") {
    pattern := '"' . RegexEscape(key) . '"\s*:\s*"((?:\\.|[^"\\])*)"'
    if !RegExMatch(json, pattern, &match) {
        return defaultValue
    }

    return JsonUnescape(match[1])
}


JsonGetNumber(json, key, defaultValue := 0) {
    pattern := '"' . RegexEscape(key) . '"\s*:\s*(-?\d+(?:\.\d+)?)'
    if !RegExMatch(json, pattern, &match) {
        return defaultValue
    }

    return match[1] + 0
}


JsonGetBoolean(json, key, defaultValue := false) {
    pattern := '"' . RegexEscape(key) . '"\s*:\s*(true|false)'
    if !RegExMatch(json, pattern, &match) {
        return defaultValue
    }

    return match[1] = "true"
}


RegexEscape(value) {
    return RegExReplace(value, '([\\\.\^\$\|\?\*\+\(\)\[\]\{\}])', '\\$1')
}


JsonUnescape(value) {
    result := ""
    index := 1

    while index <= StrLen(value) {
        character := SubStr(value, index, 1)
        if character != "\" {
            result .= character
            index += 1
            continue
        }

        index += 1
        escapeCode := SubStr(value, index, 1)

        switch escapeCode {
            case '"':
                result .= '"'
            case "\\":
                result .= "\\"
            case "/":
                result .= "/"
            case "b":
                result .= Chr(8)
            case "f":
                result .= Chr(12)
            case "n":
                result .= "`n"
            case "r":
                result .= "`r"
            case "t":
                result .= "`t"
            case "u":
                hexValue := SubStr(value, index + 1, 4)
                result .= Chr("0x" . hexValue)
                index += 4
            default:
                result .= escapeCode
        }

        index += 1
    }

    return result
}