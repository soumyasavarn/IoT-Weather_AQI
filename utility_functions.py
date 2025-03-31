import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_weather_alert(alert_type, message):
    """
    INPUT: (str, str)
    OUTPUT: None (sends an email to multiple recipients in a single email)
    """
    sender_email = "soumyasavarn2@gmail.com"
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_password = "your_app_password"  # Replace with your Gmail app password (CONFIDENTIAL)
    
    # List of recipients
    recipients = ["s.savarn@iitg.ac.in", "g.ishan@iitg.ac.in", "m.saptarshi@iitg.ac.in","s.rishab@iitg.ac.in", "m.vikky@iitg.ac.in","arghyadip@iitg.ac.in"]
    
    subject = f"Weather Alert: {alert_type}"
    
    email_body = f"""
    Hi Everyone,
    
    This is an important weather alert regarding {alert_type}:
    
    {message}
    
    Stay safe and take necessary precautions.
    
    Regards,
    Automated Weather Monitoring System
    """
    
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = ", ".join(recipients)  # Send to all in one email
    message["Subject"] = subject
    message.attach(MIMEText(email_body, "plain"))
    
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(message)
        
        print(f"Weather alert email sent to: {', '.join(recipients)}")
    
    except Exception as e:
        print(f"Error sending email: {str(e)}")

# Example Usage:
# send_weather_alert("Heavy Rainfall", "Expect heavy rains in Guwahati today. Stay indoors.")
