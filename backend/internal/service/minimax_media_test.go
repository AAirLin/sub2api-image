package service

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestBuildMiniMaxMediaURL(t *testing.T) {
	tests := []struct {
		name        string
		credentials map[string]any
		endpoint    string
		want        string
	}{
		{
			name:        "default international endpoint",
			credentials: map[string]any{},
			endpoint:    miniMaxImageGenerationEndpoint,
			want:        "https://api.minimax.io/v1/image_generation",
		},
		{
			name:        "derive media origin from anthropic base",
			credentials: map[string]any{"base_url": "https://api.minimaxi.com/anthropic"},
			endpoint:    miniMaxSpeechEndpoint,
			want:        "https://api.minimaxi.com/v1/t2a_v2",
		},
		{
			name:        "prefer explicit media base",
			credentials: map[string]any{"base_url": "https://relay.example/anthropic", "media_base_url": "https://media.example/api"},
			endpoint:    miniMaxImageGenerationEndpoint,
			want:        "https://media.example/api/v1/image_generation",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			account := &Account{Platform: PlatformMiniMax, Type: AccountTypeAPIKey, Credentials: tt.credentials}
			got, err := buildMiniMaxMediaURL(account, tt.endpoint)
			require.NoError(t, err)
			require.Equal(t, tt.want, got)
		})
	}
}

func TestBuildMiniMaxImageRequest(t *testing.T) {
	parsed := &OpenAIImagesRequest{
		Model:          "image-01",
		Prompt:         "a red paper kite",
		N:              2,
		Size:           "1792x1024",
		ResponseFormat: "b64_json",
	}
	body := []byte(`{"model":"image-01","prompt":"a red paper kite","n":2,"size":"1792x1024","response_format":"b64_json","prompt_optimizer":false,"aigc_watermark":true}`)

	encoded, err := buildMiniMaxImageRequest(parsed, body, "image-01")
	require.NoError(t, err)

	var got map[string]any
	require.NoError(t, json.Unmarshal(encoded, &got))
	require.Equal(t, "image-01", got["model"])
	require.Equal(t, "a red paper kite", got["prompt"])
	require.Equal(t, float64(2), got["n"])
	require.Equal(t, "16:9", got["aspect_ratio"])
	require.Equal(t, "base64", got["response_format"])
	require.Equal(t, false, got["prompt_optimizer"])
	require.Equal(t, true, got["aigc_watermark"])
}

func TestTransformMiniMaxImageResponse(t *testing.T) {
	body := []byte(`{
		"id":"img-request-1",
		"data":{"image_urls":["https://cdn.example/one.png"],"image_base64":["aGVsbG8="]},
		"metadata":{"billing":{"units":2}},
		"base_resp":{"status_code":0,"status_msg":"success"}
	}`)

	encoded, count, requestID, err := transformMiniMaxImageResponse(body, 1720000000)
	require.NoError(t, err)
	require.Equal(t, 2, count)
	require.Equal(t, "img-request-1", requestID)

	var got struct {
		Created  int64            `json:"created"`
		Data     []map[string]any `json:"data"`
		Metadata map[string]any   `json:"metadata"`
	}
	require.NoError(t, json.Unmarshal(encoded, &got))
	require.Equal(t, int64(1720000000), got.Created)
	require.Equal(t, "https://cdn.example/one.png", got.Data[0]["url"])
	require.Equal(t, "aGVsbG8=", got.Data[1]["b64_json"])
	require.Contains(t, got.Metadata, "billing")
}

func TestTransformMiniMaxImageResponseReturnsNativeError(t *testing.T) {
	body := []byte(`{"base_resp":{"status_code":1008,"status_msg":"insufficient balance"}}`)

	_, _, _, err := transformMiniMaxImageResponse(body, 1720000000)
	require.Error(t, err)
	var upstreamErr *MiniMaxAPIError
	require.ErrorAs(t, err, &upstreamErr)
	require.Equal(t, int64(1008), upstreamErr.Code)
	require.Equal(t, "insufficient balance", upstreamErr.Message)
}

func TestBuildMiniMaxSpeechRequest(t *testing.T) {
	body := []byte(`{
		"model":"speech-2.8-hd",
		"input":"Sub2API voice check",
		"voice":"male-qn-qingse",
		"response_format":"mp3",
		"speed":1.25,
		"metadata":{"language_boost":"Chinese","voice_setting":{"emotion":"happy"}}
	}`)

	request, err := parseMiniMaxSpeechRequest(body)
	require.NoError(t, err)
	encoded, err := buildMiniMaxSpeechRequest(request, "speech-2.8-hd")
	require.NoError(t, err)

	var got map[string]any
	require.NoError(t, json.Unmarshal(encoded, &got))
	require.Equal(t, "speech-2.8-hd", got["model"])
	require.Equal(t, "Sub2API voice check", got["text"])
	require.Equal(t, "Chinese", got["language_boost"])
	require.Equal(t, "hex", got["output_format"])
	require.Equal(t, "mp3", got["audio_setting"].(map[string]any)["format"])
	voice := got["voice_setting"].(map[string]any)
	require.Equal(t, "male-qn-qingse", voice["voice_id"])
	require.Equal(t, 1.25, voice["speed"])
	require.Equal(t, "happy", voice["emotion"])
}

func TestDecodeMiniMaxSpeechResponse(t *testing.T) {
	body := []byte(`{
		"data":{"audio":"49443303000000","status":2},
		"extra_info":{"usage_characters":19},
		"trace_id":"trace-1",
		"base_resp":{"status_code":0,"status_msg":"success"}
	}`)

	result, err := decodeMiniMaxSpeechResponse(body, "mp3")
	require.NoError(t, err)
	require.Equal(t, []byte{0x49, 0x44, 0x33, 0x03, 0x00, 0x00, 0x00}, result.Audio)
	require.Equal(t, "audio/mpeg", result.ContentType)
	require.Equal(t, 19, result.UsageCharacters)
	require.Equal(t, "trace-1", result.RequestID)
}

func TestDecodeMiniMaxSpeechResponseReturnsNativeError(t *testing.T) {
	body := []byte(`{"base_resp":{"status_code":1004,"status_msg":"invalid api key"}}`)

	_, err := decodeMiniMaxSpeechResponse(body, "mp3")
	require.Error(t, err)
	var upstreamErr *MiniMaxAPIError
	require.ErrorAs(t, err, &upstreamErr)
	require.Equal(t, int64(1004), upstreamErr.Code)
}

func TestForwardMiniMaxImagesUsesNativeEndpointAndReturnsOpenAIResponse(t *testing.T) {
	gin.SetMode(gin.TestMode)
	body := []byte(`{"model":"image-01","prompt":"draw a red kite","size":"1024x1024","response_format":"url"}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/images/generations", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = req

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body: io.NopCloser(strings.NewReader(`{
			"id":"minimax-image-request",
			"data":{"image_urls":["https://cdn.example/kite.png"]},
			"base_resp":{"status_code":0,"status_msg":"success"}
		}`)),
	}}
	svc := &OpenAIGatewayService{cfg: &config.Config{}, httpUpstream: upstream}
	parsed, err := svc.ParseOpenAIImagesRequest(c, body)
	require.NoError(t, err)
	account := &Account{
		ID: 9, Name: "minimax-images", Platform: PlatformMiniMax, Type: AccountTypeAPIKey,
		Credentials: map[string]any{"api_key": "minimax-key", "base_url": "https://api.minimaxi.com/anthropic"},
	}

	result, err := svc.ForwardImages(context.Background(), c, account, body, parsed, "")
	require.NoError(t, err)
	require.Equal(t, 1, result.ImageCount)
	require.Equal(t, "image-01", result.UpstreamModel)
	require.Equal(t, miniMaxImageGenerationEndpoint, result.UpstreamEndpoint)
	require.Equal(t, "https://api.minimaxi.com/v1/image_generation", upstream.lastReq.URL.String())
	require.Equal(t, "Bearer minimax-key", upstream.lastReq.Header.Get("Authorization"))
	require.Equal(t, "1:1", gjson.GetBytes(upstream.lastBody, "aspect_ratio").String())
	require.Equal(t, "https://cdn.example/kite.png", gjson.Get(recorder.Body.String(), "data.0.url").String())
}

func TestForwardMiniMaxImagesUsageLimitReturnsFailoverBeforeWriting(t *testing.T) {
	gin.SetMode(gin.TestMode)
	body := []byte(`{"model":"image-01","prompt":"draw a red kite","size":"1024x1024","response_format":"url"}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/images/generations", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = req

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body: io.NopCloser(strings.NewReader(`{
			"base_resp":{"status_code":2056,"status_msg":"token plan usage limit reached"}
		}`)),
	}}
	svc := &OpenAIGatewayService{cfg: &config.Config{}, httpUpstream: upstream}
	parsed, err := svc.ParseOpenAIImagesRequest(c, body)
	require.NoError(t, err)
	account := &Account{
		ID: 11, Name: "minimax-images-exhausted", Platform: PlatformMiniMax, Type: AccountTypeAPIKey,
		Credentials: map[string]any{"api_key": "minimax-key", "base_url": "https://api.minimaxi.com/anthropic"},
	}

	result, err := svc.ForwardImages(context.Background(), c, account, body, parsed, "")
	require.Nil(t, result)
	var failoverErr *UpstreamFailoverError
	require.ErrorAs(t, err, &failoverErr)
	require.Equal(t, http.StatusTooManyRequests, failoverErr.StatusCode)
	require.Contains(t, string(failoverErr.ResponseBody), `"status_code":2056`)
	require.Equal(t, 0, recorder.Body.Len())
}

func TestForwardMiniMaxSpeechReturnsDecodedAudioAndCharacterUsage(t *testing.T) {
	gin.SetMode(gin.TestMode)
	body := []byte(`{"model":"speech-2.8-hd","input":"hello","voice":"male-qn-qingse","response_format":"mp3"}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/audio/speech", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	recorder := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(recorder)
	c.Request = req

	upstream := &httpUpstreamRecorder{resp: &http.Response{
		StatusCode: http.StatusOK,
		Header:     http.Header{"Content-Type": []string{"application/json"}},
		Body: io.NopCloser(strings.NewReader(`{
			"data":{"audio":"49443303000000","status":2},
			"extra_info":{"usage_characters":5},
			"trace_id":"minimax-speech-request",
			"base_resp":{"status_code":0,"status_msg":"success"}
		}`)),
	}}
	svc := &OpenAIGatewayService{cfg: &config.Config{}, httpUpstream: upstream}
	parsed, err := ParseMiniMaxSpeechRequest(body)
	require.NoError(t, err)
	account := &Account{
		ID: 10, Name: "minimax-speech", Platform: PlatformMiniMax, Type: AccountTypeAPIKey,
		Credentials: map[string]any{"api_key": "minimax-key", "media_base_url": "https://api.minimaxi.com"},
	}

	result, err := svc.ForwardMiniMaxSpeech(context.Background(), c, account, parsed, "")
	require.NoError(t, err)
	require.Equal(t, 5, result.Usage.InputTokens)
	require.Equal(t, "minimax-speech-request", result.RequestID)
	require.Equal(t, miniMaxSpeechEndpoint, result.UpstreamEndpoint)
	require.Equal(t, "https://api.minimaxi.com/v1/t2a_v2", upstream.lastReq.URL.String())
	require.Equal(t, "hex", gjson.GetBytes(upstream.lastBody, "output_format").String())
	require.Equal(t, "audio/mpeg", recorder.Header().Get("Content-Type"))
	require.Equal(t, []byte{0x49, 0x44, 0x33, 0x03, 0x00, 0x00, 0x00}, recorder.Body.Bytes())
}

func TestMiniMaxMediaSchedulerSelectsMiniMaxAccount(t *testing.T) {
	resetOpenAIAdvancedSchedulerSettingCacheForTest()
	groupID := int64(88)
	account := Account{
		ID: 88, Name: "minimax-media", Platform: PlatformMiniMax, Type: AccountTypeAPIKey,
		Status: StatusActive, Schedulable: true, Concurrency: 2,
		Credentials: map[string]any{"api_key": "minimax-key"},
	}
	cfg := &config.Config{}
	cfg.Gateway.Scheduling.LoadBatchEnabled = false
	svc := &OpenAIGatewayService{
		accountRepo:        schedulerTestOpenAIAccountRepo{accounts: []Account{account}},
		cache:              &schedulerTestGatewayCache{},
		cfg:                cfg,
		concurrencyService: NewConcurrencyService(schedulerTestConcurrencyCache{}),
	}

	imageSelection, _, err := svc.SelectAccountWithSchedulerForImages(
		context.Background(), &groupID, "", "image-01", nil,
		OpenAIImagesCapabilityNative, PlatformMiniMax,
	)
	require.NoError(t, err)
	require.NotNil(t, imageSelection)
	require.Equal(t, account.ID, imageSelection.Account.ID)
	if imageSelection.ReleaseFunc != nil {
		imageSelection.ReleaseFunc()
	}

	speechSelection, _, err := svc.SelectAccountWithSchedulerForCapability(
		context.Background(), &groupID, "", "", "speech-2.8-hd", nil,
		OpenAIUpstreamTransportHTTPSSE, "", false, false, false, PlatformMiniMax,
	)
	require.NoError(t, err)
	require.NotNil(t, speechSelection)
	require.Equal(t, account.ID, speechSelection.Account.ID)
	if speechSelection.ReleaseFunc != nil {
		speechSelection.ReleaseFunc()
	}
}
