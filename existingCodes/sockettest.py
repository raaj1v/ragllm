import socketio
import json

# Create a new Socket.IO client instance
sio = socketio.Client()

# Define event handlers
@sio.event
def connect():
    print("Connected to the server.")
    # After connecting, emit a message to process the PDF
    tcno = '74137697'  # Example tc_no, change to the actual one you want to test
    tabId = 'tab1'  # Example tabId, change accordingly
    sio.emit('process_pdf', {'tc_no': tcno, 'tabId': tabId})

@sio.event
def connection_established(data):
    print(data)  # Output message when connection is established

@sio.event
def response(data):
    print(f"Received response: {data}")  # Output received response from server

@sio.event
def pdf_processed_complete(data):
    print(f"Processing complete: {data}")  # Indication that processing is finished

@sio.event
def pdf_processed_error(data):
    print(f"Error during processing: {data}")  # In case there's an error

@sio.event
def disconnect():
    print("Disconnected from the server.")

# Connect to the server
def run_test():
    server_url = 'http://localhost:5003'  # Adjust if necessary
    sio.connect(server_url)

    # Wait for events (you can add a timeout here if needed)
    sio.wait()

if __name__ == '__main__':
    run_test()
