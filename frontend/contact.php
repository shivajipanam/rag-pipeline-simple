<?php
/**
 * Contact Sales mailer for Volo AI.
 * Receives a POST from the landing page form and sends an email to
 * shivajipanam@gmail.com.
 *
 * Responds with plain text:
 *   "ok"    — on success
 *   error message string — on failure (HTTP 500)
 */

$to = "shivajipanam@gmail.com";

// Sanitise inputs
function clean(string $val): string {
    return htmlspecialchars(strip_tags(trim($val)), ENT_QUOTES, 'UTF-8');
}

$name    = clean($_POST['name']    ?? '');
$email   = clean($_POST['email']   ?? '');
$company = clean($_POST['company'] ?? '');
$message = clean($_POST['message'] ?? '');

// Basic validation
if (empty($name) || empty($email) || empty($message)) {
    http_response_code(400);
    echo "Please fill in all required fields.";
    exit;
}

if (!filter_var($_POST['email'], FILTER_VALIDATE_EMAIL)) {
    http_response_code(400);
    echo "Invalid email address.";
    exit;
}

// Build email
$subject = "Volo AI — Sales enquiry from {$name}" . ($company ? " ({$company})" : "");

$body  = "You have a new sales enquiry via the Volo AI landing page.\n\n";
$body .= "Name:    {$name}\n";
$body .= "Email:   {$email}\n";
$body .= "Company: " . ($company ?: "—") . "\n\n";
$body .= "Message:\n{$message}\n";

$headers  = "From: noreply@voloai.com\r\n";
$headers .= "Reply-To: {$email}\r\n";
$headers .= "X-Mailer: PHP/" . phpversion() . "\r\n";
$headers .= "MIME-Version: 1.0\r\n";
$headers .= "Content-Type: text/plain; charset=UTF-8\r\n";

if (mail($to, $subject, $body, $headers)) {
    echo "ok";
} else {
    http_response_code(500);
    echo "Failed to send email. Please try again or email us directly at shivajipanam@gmail.com";
}
