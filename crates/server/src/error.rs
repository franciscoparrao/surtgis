//! Error type mapped onto HTTP status codes.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

/// Everything a request can fail with.
#[derive(Debug, thiserror::Error)]
pub enum ServeError {
    /// Malformed request (parameters, tile address).
    #[error("{0}")]
    BadRequest(String),
    /// Source rejected by the allowlist / root policy.
    #[error("{0}")]
    Forbidden(String),
    /// Source does not exist.
    #[error("{0}")]
    NotFound(String),
    /// The requested window lies outside the source; tiles answer with
    /// a transparent image rather than an error.
    #[error("window outside the source")]
    Outside,
    /// The source could not be opened or read.
    #[error("{0}")]
    Source(String),
    /// Algorithm or rendering failure.
    #[error("{0}")]
    Compute(String),
}

impl ServeError {
    /// HTTP status for the error.
    pub fn status(&self) -> StatusCode {
        match self {
            ServeError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ServeError::Forbidden(_) => StatusCode::FORBIDDEN,
            ServeError::NotFound(_) => StatusCode::NOT_FOUND,
            ServeError::Outside => StatusCode::NO_CONTENT,
            ServeError::Source(_) => StatusCode::BAD_GATEWAY,
            ServeError::Compute(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for ServeError {
    fn into_response(self) -> Response {
        let status = self.status();
        if status == StatusCode::NO_CONTENT {
            return status.into_response();
        }
        let body = serde_json::json!({ "error": self.to_string() });
        (status, axum::Json(body)).into_response()
    }
}
