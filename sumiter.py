from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.service import Service as EdgeService
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import random
import time

# Path to Edge WebDriver
EDGE_DRIVER_PATH = "C:/WebDrivers/msedgedriver.exe"

# Google Form URL
FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSflLsr33LqZt5rjxJmkOb5wLJjzsqSh2UNCTkFkOiWG_DMaDA/viewform"

# Start WebDriver
service = EdgeService(EDGE_DRIVER_PATH)
options = webdriver.EdgeOptions()
driver = webdriver.Edge(service=service, options=options)

try:
    driver.get(FORM_URL)  # Open the form
    print("Filling out form...")

    # --- Wait for form to load ---
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

    # --- Fill First Page ---
    input_boxes = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.TAG_NAME, "input"))
    )

    if len(input_boxes) < 4:
        print("Error: Not enough input fields found.")
    else:
        age = str(random.randint(40, 49))  # Randomize age (40-49)
        input_boxes[1].send_keys(age)  # Age input
        input_boxes[2].send_keys("Fisherman")  # Job input
        print(f"Entered Age: {age}, Job: Fisherman")

    # --- Select "Kabulusan" from Dropdown ---
    try:
        dropdown = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "jgvuAb"))
        )
        dropdown.click()  # Expand dropdown

        # Wait for options to appear
        option = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//div[@data-value='Kabulusan']"))
        )
        driver.execute_script("arguments[0].click();", option)  # Click via JavaScript
        print("Successfully selected 'Kabulusan'")
    except Exception as e:
        print("Dropdown selection failed:", e)

    # Click "Next" button to go to the second page
    next_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//span[text()='Next']"))
    )
    driver.execute_script("arguments[0].click();", next_button)
    print("First page completed! Moving to Page 2...")

    # --- Fill Second Page ---
    time.sleep(2)  # Short wait for transition
    radio_buttons = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CLASS_NAME, "docssharedWizToggleLabeledContainer"))
    )

    # Select random values for each question
    for i in range(5):
        try:
            choice = random.choice(radio_buttons[i * 3:(i + 1) * 3])  # Randomize choice
            driver.execute_script("arguments[0].click();", choice)
            print(f"Selected choice for Q{i+1}")
        except:
            print(f"Failed to select choice for Q{i+1}")

    # Click "Next" to go to the third page
    next_button = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//span[text()='Next']"))
    )
    driver.execute_script("arguments[0].click();", next_button)
    print("Second page completed! Moving to Page 3...")

    # --- Fill Third Page (if needed) ---
    time.sleep(2)  # Allow page to load
    final_submit = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//span[text()='Submit']"))
    )

    # Click Submit
    driver.execute_script("arguments[0].click();", final_submit)
    print("Final page completed! Form successfully submitted.")

    # --- Wait before closing to observe results ---
    time.sleep(3)

except Exception as e:
    print("Error:", e)

finally:
    driver.quit()
    print("All steps completed.")
