import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import pygame 
import time

def play_alert_sound():
    """
    INPUT: NONE
    OUTPUT: None (plays an alert sound for 5 seconds in parallel thread without stalling the main program)
    """
    def play_sound():
        pygame.mixer.init()
        pygame.mixer.music.load("alert.mp3")
        pygame.mixer.music.play()
        time.sleep(5)  # Play for 5 seconds
        pygame.mixer.music.stop()
    
    threading.Thread(target=play_sound).start()
    print ("alert played")



def send_weather_alert(alert_type, message):
    """
    INPUT: (str, str)
    OUTPUT: None (sends an email to multiple recipients in a single email)
    """
    sender_email = "soumyasavarn2@gmail.com"
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_password = "blmj huad sqan dbnf"  # Replace with your Gmail app password (CONFIDENTIAL)
    
    # List of recipients
    recipients = ["s.savagsrn@iitg.ac.in"] # for testing

    # Final list of recipients (uncomment for actual use)
    # recipients = ["s.savarn@iitg.ac.in", "g.ishan@iitg.ac.in", "m.saptarshi@iitg.ac.in","s.rishab@iitg.ac.in", "m.vikky@iitg.ac.in","arghyadip@iitg.ac.in"]
    
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
# play_alert_sound()