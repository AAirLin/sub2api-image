package admin

import (
	"testing"

	"github.com/gin-gonic/gin/binding"
	"github.com/stretchr/testify/require"
)

func TestGroupRequestsAcceptMiniMaxPlatform(t *testing.T) {
	require.NoError(t, binding.Validator.ValidateStruct(&CreateGroupRequest{
		Name:     "MiniMax media",
		Platform: "minimax",
	}))
	require.NoError(t, binding.Validator.ValidateStruct(&UpdateGroupRequest{
		Platform: "minimax",
	}))
}
