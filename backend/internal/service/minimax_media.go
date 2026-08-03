package service

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/minimax"
	"github.com/Wei-Shaw/sub2api/internal/util/responseheaders"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
)

const (
	miniMaxImageGenerationEndpoint = "/v1/image_generation"
	miniMaxSpeechEndpoint          = "/v1/t2a_v2"
)

// MiniMaxAPIError is a provider-level error returned inside an otherwise valid JSON response.
type MiniMaxAPIError struct {
	Code    int64
	Message string
}

func (e *MiniMaxAPIError) Error() string {
	if e == nil {
		return ""
	}
	return fmt.Sprintf("minimax api error %d: %s", e.Code, strings.TrimSpace(e.Message))
}

type miniMaxBaseResponse struct {
	StatusCode int64  `json:"status_code"`
	StatusMsg  string `json:"status_msg"`
}

type miniMaxImageRequest struct {
	Model           string `json:"model"`
	Prompt          string `json:"prompt"`
	AspectRatio     string `json:"aspect_ratio,omitempty"`
	ResponseFormat  string `json:"response_format,omitempty"`
	N               int    `json:"n,omitempty"`
	PromptOptimizer *bool  `json:"prompt_optimizer,omitempty"`
	AIGCWatermark   *bool  `json:"aigc_watermark,omitempty"`
}

type miniMaxImageResponse struct {
	ID   string `json:"id"`
	Data struct {
		ImageURLs   []string `json:"image_urls"`
		ImageBase64 []string `json:"image_base64"`
	} `json:"data"`
	Metadata map[string]any      `json:"metadata"`
	BaseResp miniMaxBaseResponse `json:"base_resp"`
}

// MiniMaxSpeechRequest is the OpenAI-compatible speech request accepted by the gateway.
type MiniMaxSpeechRequest struct {
	Model          string          `json:"model"`
	Input          string          `json:"input"`
	Voice          string          `json:"voice"`
	ResponseFormat string          `json:"response_format"`
	Speed          *float64        `json:"speed,omitempty"`
	Metadata       json.RawMessage `json:"metadata,omitempty"`
}

type miniMaxSpeechUpstreamRequest struct {
	Model             string                    `json:"model"`
	Text              string                    `json:"text"`
	Stream            bool                      `json:"stream,omitempty"`
	VoiceSetting      miniMaxVoiceSetting       `json:"voice_setting"`
	PronunciationDict *miniMaxPronunciationDict `json:"pronunciation_dict,omitempty"`
	AudioSetting      *miniMaxAudioSetting      `json:"audio_setting,omitempty"`
	TimbreWeights     []miniMaxTimbreWeight     `json:"timbre_weights,omitempty"`
	LanguageBoost     string                    `json:"language_boost,omitempty"`
	VoiceModify       *miniMaxVoiceModify       `json:"voice_modify,omitempty"`
	SubtitleEnable    bool                      `json:"subtitle_enable,omitempty"`
	OutputFormat      string                    `json:"output_format,omitempty"`
	AIGCWatermark     bool                      `json:"aigc_watermark,omitempty"`
}

type miniMaxVoiceSetting struct {
	VoiceID           string  `json:"voice_id"`
	Speed             float64 `json:"speed,omitempty"`
	Volume            float64 `json:"vol,omitempty"`
	Pitch             int     `json:"pitch,omitempty"`
	Emotion           string  `json:"emotion,omitempty"`
	TextNormalization bool    `json:"text_normalization,omitempty"`
	LatexRead         bool    `json:"latex_read,omitempty"`
}

type miniMaxPronunciationDict struct {
	Tone []string `json:"tone,omitempty"`
}

type miniMaxAudioSetting struct {
	SampleRate int    `json:"sample_rate,omitempty"`
	Bitrate    int    `json:"bitrate,omitempty"`
	Format     string `json:"format,omitempty"`
	Channel    int    `json:"channel,omitempty"`
	ForceCBR   bool   `json:"force_cbr,omitempty"`
}

type miniMaxTimbreWeight struct {
	VoiceID string `json:"voice_id"`
	Weight  int    `json:"weight"`
}

type miniMaxVoiceModify struct {
	Pitch        int    `json:"pitch,omitempty"`
	Intensity    int    `json:"intensity,omitempty"`
	Timbre       int    `json:"timbre,omitempty"`
	SoundEffects string `json:"sound_effects,omitempty"`
}

type miniMaxSpeechResponse struct {
	Data struct {
		Audio  string `json:"audio"`
		Status int    `json:"status"`
	} `json:"data"`
	ExtraInfo struct {
		UsageCharacters int `json:"usage_characters"`
	} `json:"extra_info"`
	TraceID  string              `json:"trace_id"`
	BaseResp miniMaxBaseResponse `json:"base_resp"`
}

type miniMaxDecodedSpeech struct {
	Audio           []byte
	RedirectURL     string
	ContentType     string
	UsageCharacters int
	RequestID       string
}

func (a *Account) GetMiniMaxMediaBaseURL() string {
	if a == nil || !a.IsMiniMax() {
		return ""
	}
	if configured := strings.TrimSpace(a.GetCredential("media_base_url")); configured != "" {
		return configured
	}
	configured := strings.TrimSpace(a.GetCredential("base_url"))
	if configured == "" {
		return strings.TrimSuffix(minimax.DefaultAnthropicBaseURL, "/anthropic")
	}

	parsed, err := url.Parse(configured)
	if err != nil {
		return configured
	}
	path := strings.TrimRight(parsed.Path, "/")
	for _, suffix := range []string{"/v1/image_generation", "/v1/t2a_v2", "/anthropic/v1", "/anthropic", "/v1"} {
		if strings.HasSuffix(strings.ToLower(path), suffix) {
			path = strings.TrimSuffix(path, path[len(path)-len(suffix):])
			break
		}
	}
	parsed.Path = strings.TrimRight(path, "/")
	parsed.RawPath = ""
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return strings.TrimRight(parsed.String(), "/")
}

func buildMiniMaxMediaURL(account *Account, endpoint string) (string, error) {
	if account == nil || !account.IsMiniMax() {
		return "", fmt.Errorf("minimax account is required")
	}
	base := strings.TrimSpace(account.GetMiniMaxMediaBaseURL())
	if base == "" {
		return "", fmt.Errorf("minimax media base URL is required")
	}
	parsed, err := url.Parse(base)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		return "", fmt.Errorf("invalid minimax media base URL")
	}
	parsed.Path = strings.TrimRight(parsed.Path, "/") + "/" + strings.TrimLeft(endpoint, "/")
	parsed.RawPath = ""
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return parsed.String(), nil
}

func buildMiniMaxImageRequest(parsed *OpenAIImagesRequest, body []byte, upstreamModel string) ([]byte, error) {
	if parsed == nil {
		return nil, fmt.Errorf("parsed images request is required")
	}
	request := miniMaxImageRequest{
		Model:          strings.TrimSpace(upstreamModel),
		Prompt:         parsed.Prompt,
		AspectRatio:    miniMaxAspectRatio(parsed.Size),
		ResponseFormat: normalizeMiniMaxImageResponseFormat(parsed.ResponseFormat),
		N:              parsed.N,
	}
	if request.Model == "" {
		request.Model = "image-01"
	}
	if request.N <= 0 {
		request.N = 1
	}
	if raw := gjson.GetBytes(body, "aspect_ratio"); raw.Type == gjson.String && strings.TrimSpace(raw.String()) != "" {
		request.AspectRatio = strings.TrimSpace(raw.String())
	}
	if raw := gjson.GetBytes(body, "prompt_optimizer"); raw.Type == gjson.True || raw.Type == gjson.False {
		value := raw.Bool()
		request.PromptOptimizer = &value
	}
	if raw := gjson.GetBytes(body, "aigc_watermark"); raw.Type == gjson.True || raw.Type == gjson.False {
		value := raw.Bool()
		request.AIGCWatermark = &value
	}
	return json.Marshal(request)
}

func normalizeMiniMaxImageResponseFormat(format string) string {
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "", "url":
		return "url"
	case "b64_json", "base64":
		return "base64"
	default:
		return strings.TrimSpace(format)
	}
}

func miniMaxAspectRatio(size string) string {
	switch strings.ToLower(strings.TrimSpace(size)) {
	case "1024x1024":
		return "1:1"
	case "1792x1024":
		return "16:9"
	case "1024x1792":
		return "9:16"
	case "1536x1024", "1248x832":
		return "3:2"
	case "1024x1536", "832x1248":
		return "2:3"
	case "1152x864":
		return "4:3"
	case "864x1152":
		return "3:4"
	case "1344x576":
		return "21:9"
	}
	parts := strings.Split(strings.ToLower(strings.TrimSpace(size)), "x")
	if len(parts) != 2 {
		return ""
	}
	width, widthErr := strconv.Atoi(parts[0])
	height, heightErr := strconv.Atoi(parts[1])
	if widthErr != nil || heightErr != nil || width <= 0 || height <= 0 {
		return ""
	}
	divisor := miniMaxGCD(width, height)
	ratio := fmt.Sprintf("%d:%d", width/divisor, height/divisor)
	switch ratio {
	case "1:1", "16:9", "4:3", "3:2", "2:3", "3:4", "9:16", "21:9":
		return ratio
	default:
		return ""
	}
}

func miniMaxGCD(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	if a == 0 {
		return 1
	}
	return a
}

func transformMiniMaxImageResponse(body []byte, created int64) ([]byte, int, string, error) {
	var response miniMaxImageResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, 0, "", fmt.Errorf("decode minimax image response: %w", err)
	}
	if response.BaseResp.StatusCode != 0 {
		return nil, 0, response.ID, &MiniMaxAPIError{Code: response.BaseResp.StatusCode, Message: response.BaseResp.StatusMsg}
	}
	data := make([]map[string]string, 0, len(response.Data.ImageURLs)+len(response.Data.ImageBase64))
	for _, imageURL := range response.Data.ImageURLs {
		data = append(data, map[string]string{"url": imageURL})
	}
	for _, imageBase64 := range response.Data.ImageBase64 {
		data = append(data, map[string]string{"b64_json": imageBase64})
	}
	payload := struct {
		Created  int64               `json:"created"`
		Data     []map[string]string `json:"data"`
		Metadata map[string]any      `json:"metadata,omitempty"`
	}{Created: created, Data: data, Metadata: response.Metadata}
	encoded, err := json.Marshal(payload)
	if err != nil {
		return nil, 0, response.ID, err
	}
	return encoded, len(data), response.ID, nil
}

func parseMiniMaxSpeechRequest(body []byte) (*MiniMaxSpeechRequest, error) {
	var request MiniMaxSpeechRequest
	if err := json.Unmarshal(body, &request); err != nil {
		return nil, fmt.Errorf("failed to parse request body")
	}
	request.Model = strings.TrimSpace(request.Model)
	request.Input = strings.TrimSpace(request.Input)
	request.Voice = strings.TrimSpace(request.Voice)
	request.ResponseFormat = strings.ToLower(strings.TrimSpace(request.ResponseFormat))
	if request.Model == "" {
		return nil, fmt.Errorf("model is required")
	}
	if request.Input == "" {
		return nil, fmt.Errorf("input is required")
	}
	if request.Voice == "" {
		return nil, fmt.Errorf("voice is required")
	}
	if request.ResponseFormat == "" {
		request.ResponseFormat = "mp3"
	}
	switch request.ResponseFormat {
	case "mp3", "wav", "flac", "aac", "pcm":
	default:
		return nil, fmt.Errorf("unsupported response_format %q", request.ResponseFormat)
	}
	if request.Speed != nil && (*request.Speed < 0.5 || *request.Speed > 2.0) {
		return nil, fmt.Errorf("speed must be between 0.5 and 2.0")
	}
	return &request, nil
}

// ParseMiniMaxSpeechRequest validates an OpenAI-compatible MiniMax speech request.
func ParseMiniMaxSpeechRequest(body []byte) (*MiniMaxSpeechRequest, error) {
	return parseMiniMaxSpeechRequest(body)
}

func buildMiniMaxSpeechRequest(request *MiniMaxSpeechRequest, upstreamModel string) ([]byte, error) {
	if request == nil {
		return nil, fmt.Errorf("speech request is required")
	}
	speed := 1.0
	if request.Speed != nil {
		speed = *request.Speed
	}
	native := miniMaxSpeechUpstreamRequest{
		Model: strings.TrimSpace(upstreamModel),
		Text:  request.Input,
		VoiceSetting: miniMaxVoiceSetting{
			VoiceID: request.Voice,
			Speed:   speed,
		},
		AudioSetting: &miniMaxAudioSetting{Format: request.ResponseFormat},
		OutputFormat: "hex",
	}
	if native.Model == "" {
		native.Model = request.Model
	}
	if len(request.Metadata) > 0 && string(request.Metadata) != "null" {
		if err := json.Unmarshal(request.Metadata, &native); err != nil {
			return nil, fmt.Errorf("invalid metadata: %w", err)
		}
	}
	// Core compatibility fields cannot be replaced by metadata. The provider-specific
	// nested options remain overridable, matching NewAPI's existing behavior.
	native.Model = strings.TrimSpace(upstreamModel)
	if native.Model == "" {
		native.Model = request.Model
	}
	native.Text = request.Input
	native.OutputFormat = "hex"
	if native.AudioSetting == nil {
		native.AudioSetting = &miniMaxAudioSetting{}
	}
	if native.AudioSetting.Format == "" {
		native.AudioSetting.Format = request.ResponseFormat
	}
	if native.VoiceSetting.VoiceID == "" {
		native.VoiceSetting.VoiceID = request.Voice
	}
	return json.Marshal(native)
}

func decodeMiniMaxSpeechResponse(body []byte, format string) (*miniMaxDecodedSpeech, error) {
	var response miniMaxSpeechResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("decode minimax speech response: %w", err)
	}
	if response.BaseResp.StatusCode != 0 {
		return nil, &MiniMaxAPIError{Code: response.BaseResp.StatusCode, Message: response.BaseResp.StatusMsg}
	}
	encodedAudio := strings.TrimSpace(response.Data.Audio)
	if encodedAudio == "" {
		return nil, fmt.Errorf("minimax speech response contains no audio")
	}
	result := &miniMaxDecodedSpeech{
		ContentType:     miniMaxSpeechContentType(format),
		UsageCharacters: response.ExtraInfo.UsageCharacters,
		RequestID:       response.TraceID,
	}
	if strings.HasPrefix(strings.ToLower(encodedAudio), "http://") || strings.HasPrefix(strings.ToLower(encodedAudio), "https://") {
		result.RedirectURL = encodedAudio
		return result, nil
	}
	audio, err := hex.DecodeString(encodedAudio)
	if err != nil {
		return nil, fmt.Errorf("decode minimax speech audio: %w", err)
	}
	result.Audio = audio
	return result, nil
}

func miniMaxSpeechContentType(format string) string {
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "wav":
		return "audio/wav"
	case "flac":
		return "audio/flac"
	case "aac":
		return "audio/aac"
	case "pcm":
		return "audio/pcm"
	default:
		return "audio/mpeg"
	}
}

func (s *OpenAIGatewayService) buildMiniMaxMediaRequest(ctx context.Context, c *gin.Context, account *Account, body []byte, endpoint string) (*http.Request, error) {
	targetURL, err := buildMiniMaxMediaURL(account, endpoint)
	if err != nil {
		return nil, err
	}
	validatedURL, err := s.validateUpstreamBaseURL(targetURL)
	if err != nil {
		return nil, err
	}
	apiKey := strings.TrimSpace(account.GetCredential("api_key"))
	if apiKey == "" {
		return nil, fmt.Errorf("api_key not found in credentials")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, validatedURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req = req.WithContext(WithHTTPUpstreamProfile(req.Context(), HTTPUpstreamProfileOpenAI))
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	if c != nil && c.Request != nil {
		if userAgent := strings.TrimSpace(c.GetHeader("User-Agent")); userAgent != "" {
			req.Header.Set("User-Agent", userAgent)
		}
	}
	account.ApplyHeaderOverrides(req.Header)
	return req, nil
}

func (s *OpenAIGatewayService) forwardMiniMaxRequest(ctx context.Context, c *gin.Context, account *Account, body []byte, endpoint, upstreamModel string) (*http.Response, []byte, error) {
	req, err := s.buildMiniMaxMediaRequest(ctx, c, account, body, endpoint)
	if err != nil {
		return nil, nil, err
	}
	proxyURL := ""
	if account.ProxyID != nil && account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}
	upstreamStart := time.Now()
	resp, err := s.httpUpstream.Do(req, proxyURL, account.ID, account.Concurrency)
	SetOpsLatencyMs(c, OpsUpstreamLatencyMsKey, time.Since(upstreamStart).Milliseconds())
	if err != nil {
		safeErr := sanitizeUpstreamErrorMessage(err.Error())
		setOpsUpstreamError(c, 0, safeErr, "")
		return nil, nil, fmt.Errorf("upstream request failed: %s", safeErr)
	}
	responseBody, readErr := ReadUpstreamResponseBody(resp.Body, s.cfg, c, openAITooLargeError)
	_ = resp.Body.Close()
	if readErr != nil {
		return resp, nil, readErr
	}
	resp.Body = io.NopCloser(bytes.NewReader(responseBody))
	if resp.StatusCode >= http.StatusBadRequest {
		message := sanitizeUpstreamErrorMessage(extractUpstreamErrorMessage(responseBody))
		if message == "" {
			message = strings.TrimSpace(gjson.GetBytes(responseBody, "base_resp.status_msg").String())
		}
		if s.shouldFailoverOpenAIUpstreamResponse(resp.StatusCode, message, responseBody) {
			shouldDisable := s.handleFailoverSideEffects(ctx, resp, account, responseBody, upstreamModel)
			return resp, responseBody, &UpstreamFailoverError{
				StatusCode:             resp.StatusCode,
				ResponseBody:           responseBody,
				RetryableOnSameAccount: !shouldDisable && account.IsPoolMode() && account.IsPoolModeRetryableStatus(resp.StatusCode),
			}
		}
		if message == "" {
			message = http.StatusText(resp.StatusCode)
		}
		writeMiniMaxErrorResponse(c, resp.StatusCode, "minimax_upstream_error", message, "")
		return resp, responseBody, &OpenAIImagesUpstreamError{StatusCode: resp.StatusCode, ErrorType: "minimax_upstream_error", Message: message}
	}
	return resp, responseBody, nil
}

func writeMiniMaxErrorResponse(c *gin.Context, status int, errorType, message, code string) {
	if c == nil {
		return
	}
	if status < 400 {
		status = http.StatusBadRequest
	}
	errorPayload := gin.H{"type": errorType, "message": message}
	if strings.TrimSpace(code) != "" {
		errorPayload["code"] = code
	}
	c.JSON(status, gin.H{"error": errorPayload})
}

func miniMaxNativeErrorStatus(code int64) int {
	switch code {
	case 1004:
		return http.StatusUnauthorized
	case 1008:
		return http.StatusPaymentRequired
	case 2056:
		return http.StatusTooManyRequests
	default:
		return http.StatusBadRequest
	}
}

func miniMaxNativeErrorShouldFailover(code int64) bool {
	switch code {
	case 1004, 1008, 2056:
		return true
	default:
		return false
	}
}

func (s *OpenAIGatewayService) newMiniMaxNativeFailoverError(
	ctx context.Context,
	resp *http.Response,
	account *Account,
	responseBody []byte,
	upstreamModel string,
	nativeErr *MiniMaxAPIError,
) *UpstreamFailoverError {
	status := miniMaxNativeErrorStatus(nativeErr.Code)
	responseHeaders := http.Header(nil)
	if resp != nil {
		responseHeaders = resp.Header.Clone()
		syntheticResp := *resp
		syntheticResp.StatusCode = status
		s.handleFailoverSideEffects(ctx, &syntheticResp, account, responseBody, upstreamModel)
	}
	return &UpstreamFailoverError{
		StatusCode:        status,
		ResponseBody:      responseBody,
		ResponseHeaders:   responseHeaders,
		Scope:             GatewayFailureScopeAccount,
		NextAccountAction: NextAccountRetry,
	}
}

func (s *OpenAIGatewayService) forwardMiniMaxImages(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
	parsed *OpenAIImagesRequest,
	channelMappedModel string,
) (*OpenAIForwardResult, error) {
	if parsed.IsEdits() {
		return nil, fmt.Errorf("MiniMax does not support image edits")
	}
	startTime := time.Now()
	requestModel := strings.TrimSpace(parsed.Model)
	if mapped := strings.TrimSpace(channelMappedModel); mapped != "" {
		requestModel = mapped
	}
	upstreamModel := account.GetMappedModel(requestModel)
	forwardBody, err := buildMiniMaxImageRequest(parsed, body, upstreamModel)
	if err != nil {
		return nil, err
	}
	resp, responseBody, err := s.forwardMiniMaxRequest(ctx, c, account, forwardBody, miniMaxImageGenerationEndpoint, upstreamModel)
	if err != nil {
		return nil, err
	}
	transformed, imageCount, requestID, err := transformMiniMaxImageResponse(responseBody, startTime.Unix())
	if err != nil {
		var nativeErr *MiniMaxAPIError
		if errors.As(err, &nativeErr) {
			if miniMaxNativeErrorShouldFailover(nativeErr.Code) {
				return nil, s.newMiniMaxNativeFailoverError(ctx, resp, account, responseBody, upstreamModel, nativeErr)
			}
			status := miniMaxNativeErrorStatus(nativeErr.Code)
			writeMiniMaxErrorResponse(c, status, "minimax_image_error", nativeErr.Message, strconv.FormatInt(nativeErr.Code, 10))
			return nil, &OpenAIImagesUpstreamError{StatusCode: status, ErrorType: "minimax_image_error", Code: strconv.FormatInt(nativeErr.Code, 10), Message: nativeErr.Message}
		}
		return nil, err
	}
	responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
	c.Data(http.StatusOK, "application/json", transformed)
	return &OpenAIForwardResult{
		RequestID:        firstNonEmpty(requestID, resp.Header.Get("x-request-id")),
		Usage:            OpenAIUsage{},
		Model:            requestModel,
		UpstreamModel:    upstreamModel,
		UpstreamEndpoint: miniMaxImageGenerationEndpoint,
		ResponseHeaders:  resp.Header.Clone(),
		Duration:         time.Since(startTime),
		ImageCount:       imageCount,
		ImageSize:        parsed.SizeTier,
		ImageInputSize:   parsed.Size,
	}, nil
}

// ForwardMiniMaxSpeech forwards an OpenAI-compatible speech request to MiniMax.
func (s *OpenAIGatewayService) ForwardMiniMaxSpeech(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	request *MiniMaxSpeechRequest,
	channelMappedModel string,
) (*OpenAIForwardResult, error) {
	if account == nil || !account.IsMiniMax() || account.Type != AccountTypeAPIKey {
		return nil, fmt.Errorf("MiniMax API key account is required")
	}
	startTime := time.Now()
	requestModel := strings.TrimSpace(request.Model)
	if mapped := strings.TrimSpace(channelMappedModel); mapped != "" {
		requestModel = mapped
	}
	upstreamModel := account.GetMappedModel(requestModel)
	forwardBody, err := buildMiniMaxSpeechRequest(request, upstreamModel)
	if err != nil {
		return nil, err
	}
	resp, responseBody, err := s.forwardMiniMaxRequest(ctx, c, account, forwardBody, miniMaxSpeechEndpoint, upstreamModel)
	if err != nil {
		return nil, err
	}
	decoded, err := decodeMiniMaxSpeechResponse(responseBody, request.ResponseFormat)
	if err != nil {
		var nativeErr *MiniMaxAPIError
		if errors.As(err, &nativeErr) {
			if miniMaxNativeErrorShouldFailover(nativeErr.Code) {
				return nil, s.newMiniMaxNativeFailoverError(ctx, resp, account, responseBody, upstreamModel, nativeErr)
			}
			status := miniMaxNativeErrorStatus(nativeErr.Code)
			writeMiniMaxErrorResponse(c, status, "minimax_speech_error", nativeErr.Message, strconv.FormatInt(nativeErr.Code, 10))
			return nil, &OpenAIImagesUpstreamError{StatusCode: status, ErrorType: "minimax_speech_error", Code: strconv.FormatInt(nativeErr.Code, 10), Message: nativeErr.Message}
		}
		return nil, err
	}
	responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
	if decoded.RedirectURL != "" {
		c.Redirect(http.StatusFound, decoded.RedirectURL)
	} else {
		c.Header("Content-Type", decoded.ContentType)
		c.Data(http.StatusOK, decoded.ContentType, decoded.Audio)
	}
	return &OpenAIForwardResult{
		RequestID:        firstNonEmpty(decoded.RequestID, resp.Header.Get("x-request-id")),
		Usage:            OpenAIUsage{InputTokens: decoded.UsageCharacters},
		Model:            requestModel,
		UpstreamModel:    upstreamModel,
		UpstreamEndpoint: miniMaxSpeechEndpoint,
		ResponseHeaders:  resp.Header.Clone(),
		Duration:         time.Since(startTime),
	}, nil
}
