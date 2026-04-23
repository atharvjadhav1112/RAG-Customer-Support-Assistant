"""
data/generate_sample_pdf.py
─────────────────────────────────────────────────────────────
Generates a realistic customer-support knowledge-base PDF
so you can test the system immediately without your own documents.

Run:
    python data/generate_sample_pdf.py
"""

import sys
from pathlib import Path

# ── Knowledge base content ─────────────────────────────────────────────────
CONTENT = """
TECHSHOP ELECTRONICS — CUSTOMER SUPPORT KNOWLEDGE BASE
Version 3.0 | Effective January 2025

=======================================================
SECTION 1: RETURN & REFUND POLICY
=======================================================

Return Window
All products purchased from TechShop Electronics may be returned within 30 days
of the purchase date for a full refund. Electronics accessories such as cables,
cases, and chargers have a 15-day return window. Software licenses and digital
downloads are non-refundable once the activation key has been used.

Refund Process
To initiate a refund, customers must submit a request through our website at
support.techshop.com/refunds or call 1-800-TECHSHOP (1-800-832-4746).
Refunds are processed within 5 to 7 business days to the original payment method.
Cash purchase refunds are issued as store credit.

Product Condition Requirements
Products must be returned in their original condition, including all accessories,
manuals, and original packaging. Items showing physical damage, water damage,
or evidence of unauthorised modification are not eligible for return. All returns
undergo a quality inspection before the refund is approved.

=======================================================
SECTION 2: WARRANTY INFORMATION
=======================================================

Standard Warranty Coverage
All TechShop Electronics products are covered by a 1-year limited manufacturer
warranty against defects in materials and workmanship under normal use.
Laptops and tablets are covered for 2 years. Accessories carry a 6-month warranty.

What Is Covered
The warranty covers manufacturing defects, hardware failures that occur under
normal operating conditions, display defects that appear within the first 90 days,
and battery failures where capacity drops below 80 percent within the warranty period.

What Is Not Covered
The warranty does not cover physical damage from drops or impacts, liquid damage,
damage from unauthorised repairs or modifications, cosmetic damage such as scratches
or dents, theft or loss, or damage caused by incompatible third-party accessories.
Normal wear and tear, including gradual battery capacity loss beyond the first year,
is excluded from warranty claims.

TechShop Protection Plus
Customers may purchase TechShop Protection Plus within 30 days of the original
purchase. This plan extends warranty coverage to 3 years and adds accidental
damage protection. The cost is 15 percent of the product purchase price.
Accidental damage claims are limited to 2 incidents per year.

=======================================================
SECTION 3: SHIPPING & DELIVERY
=======================================================

Standard Shipping
All orders over $50 qualify for free standard shipping with a delivery window
of 5 to 7 business days. Orders below $50 are charged a flat $4.99 shipping fee.
Standard shipping is available to all 50 US states, Washington D.C., and Puerto Rico.

Express Shipping Options
Express 2-day shipping is available for $12.99 per order. Next-day delivery is
available in select metropolitan areas for $19.99. Express orders placed before
2:00 PM Eastern Time ship the same business day.

International Shipping
TechShop currently ships internationally to Canada, the United Kingdom, Australia,
and most European Union member countries. International shipping fees are calculated
at checkout based on destination and order weight, and typically take 10 to 15
business days. Customers are solely responsible for customs duties, import taxes,
and any applicable brokerage fees.

Order Tracking
Every shipped order receives a tracking number via email within 24 hours of dispatch.
Real-time tracking is available through our website and via the TechShop mobile app
available on iOS and Android.

=======================================================
SECTION 4: TECHNICAL SUPPORT
=======================================================

Support Channels
Live Chat: Available 24 hours a day, 7 days a week on our website.
Phone: 1-800-TECHSHOP, Monday through Friday 8 AM to 10 PM ET,
Saturday and Sunday 9 AM to 6 PM ET.
Email: support@techshop.com with a guaranteed response within 24 hours.
Community Forum: community.techshop.com for peer-to-peer assistance.

Device Will Not Power On
Ensure the device has been charged for at least 30 minutes. Perform a forced
restart by holding the power button for 15 seconds. If the screen remains dark,
connect the device to its original charger and wait 30 minutes before attempting
to power on again. If the issue persists after charging, contact technical support.

Wi-Fi Connectivity Problems
From the device Settings menu, select Wi-Fi, tap the network name, and choose
Forget Network. Re-scan and reconnect entering your Wi-Fi password carefully.
Ensure your router firmware is current. Keep the device within 30 feet of the
router during troubleshooting. If the device cannot detect any networks, toggle
Airplane Mode on and off, then re-test.

Bluetooth Pairing Issues
Toggle Bluetooth off and on in the device settings. Put the accessory you are
pairing into its discoverable or pairing mode as described in its manual.
On the device, tap Scan or Search for Devices and select the accessory from the
list. If pairing fails, clear the accessory from the device's paired-device list,
restart both devices, and try the pairing sequence again from the beginning.

Software Update Errors
If a software update fails, ensure the device has at least 50 percent battery
charge and is connected to a stable Wi-Fi network. Navigate to Settings, then
System, then Software Update, and select Check for Updates. If an error code is
displayed, note the exact code and contact TechShop technical support for guidance.
Do not interrupt an update in progress, as this may render the device unbootable.

=======================================================
SECTION 5: ACCOUNT & ORDER MANAGEMENT
=======================================================

Creating a TechShop Account
Visit techshop.com/register to create a free account. Account benefits include
access to full order history, faster checkout with saved addresses and payment
methods, exclusive member-only discounts, and early access to seasonal sales events.

Cancelling an Order
Orders can be cancelled within 2 hours of placement if they have not yet entered
the fulfilment queue. To cancel, log in to your account, navigate to My Orders,
select the order, and click Cancel Order. After the 2-hour window the order
cannot be cancelled; you must accept delivery and initiate a standard return.

Price Match Guarantee
TechShop will match the price of any identical product offered by a major
authorised retailer if the lower price is found within 14 days of your purchase.
Clearance prices, limited-time flash sales, and marketplace third-party sellers
are excluded from price matching. Submit your price match request with proof
of the competitor's advertised price at support.techshop.com/pricematch.

=======================================================
SECTION 6: PRIVACY, DATA & SECURITY
=======================================================

Data We Collect
TechShop collects purchase and browsing history on our website, device registration
information for warranty administration, contact details for order communication,
and usage data to improve our services. We do not sell or rent customer personal
data to third parties for marketing purposes.

Requesting Data Deletion
Customers may request deletion of their account and all associated personal data
by emailing privacy@techshop.com with the subject line Data Deletion Request.
All deletion requests are completed within 30 calendar days. Note that data
required for legal compliance, such as transaction records, may be retained for
the period required by applicable law.

Account Security Best Practices
We recommend enabling two-factor authentication (2FA) in your account security
settings. Use a unique, strong password of at least 12 characters that is not
shared with other services. TechShop support staff will never ask for your
password. If you suspect your account has been compromised, change your password
immediately and contact security@techshop.com.

=======================================================
SECTION 7: STORE LOCATIONS & IN-STORE SERVICES
=======================================================

Store Hours
The majority of TechShop retail locations are open Monday through Saturday
from 9 AM to 9 PM and Sunday from 10 AM to 7 PM. Hours may vary on public
holidays. Use the store locator at techshop.com/stores to find your nearest
location and confirm its current hours.

In-Store TechDesk Service
Every TechShop location includes a TechDesk staffed by certified technicians.
TechDesk services include device diagnostics and repairs, data migration
assistance, setup and configuration of new purchases, trade-in evaluations,
and same-day in-store pickup of online orders. TechDesk appointments can be
booked online at techshop.com/techdesk.

Trade-In Programme
TechShop accepts eligible used devices in working condition for trade-in credit.
Trade-in value is determined at the time of evaluation and applied as store credit
toward a new purchase. Trade-in values are valid for 14 days from the date of
evaluation. Devices with cracked screens, water damage, or iCloud/Google account
locks are not eligible for trade-in.
"""


def generate():
    """Write CONTENT as a plaintext-based PDF to data/sample.pdf."""
    out = Path(__file__).parent / "sample.pdf"

    # Try reportlab for a proper PDF
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet

        doc    = SimpleDocTemplate(str(out), pagesize=letter,
                                   leftMargin=inch, rightMargin=inch,
                                   topMargin=inch, bottomMargin=inch)
        styles = getSampleStyleSheet()
        story  = []

        for line in CONTENT.strip().split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 8))
            elif line.startswith("=") and len(line) > 6:
                story.append(Spacer(1, 12))
                story.append(Paragraph(f"<b>{line.strip('=').strip()}</b>",
                                        styles["Heading2"]))
            elif line[0].isupper() and not line.endswith(".") and len(line) < 60:
                story.append(Spacer(1, 6))
                story.append(Paragraph(f"<b>{line}</b>", styles["Normal"]))
            else:
                story.append(Paragraph(line, styles["Normal"]))

        doc.build(story)
        print(f"✅ Sample PDF generated (reportlab): {out}  [{out.stat().st_size:,} bytes]")
        return

    except ImportError:
        pass  # reportlab not installed → fall back

    # Minimal raw PDF fallback
    _write_raw_pdf(out)
    print(f"✅ Sample PDF generated (raw): {out}  [{out.stat().st_size:,} bytes]")


def _write_raw_pdf(path: Path):
    """Write a minimal valid PDF with embedded text using only stdlib."""
    lines = [l for l in CONTENT.strip().split("\n")]

    PAGE_H = 792
    MARGIN = 50
    LINE_H = 13
    MAX_Y  = PAGE_H - MARGIN

    pages: list[list[str]] = []
    cur_page: list[str]    = []
    y = MAX_Y

    for line in lines:
        # Split long lines every 100 chars
        while len(line) > 100:
            cur_page.append((y, line[:100]))
            line = line[100:]
            y -= LINE_H
            if y < MARGIN:
                pages.append(cur_page); cur_page = []; y = MAX_Y
        cur_page.append((y, line))
        y -= LINE_H
        if y < MARGIN:
            pages.append(cur_page); cur_page = []; y = MAX_Y

    if cur_page:
        pages.append(cur_page)

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    # Build PDF objects
    objs: list[bytes] = []

    def add(content: str) -> int:
        objs.append(content.encode())
        return len(objs)  # 1-based object number

    add("<< /Type /Catalog /Pages 2 0 R >>")           # obj 1
    add("<< /Type /Pages /Kids [] /Count 0 >>")        # obj 2 placeholder
    add("<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")  # obj 3

    page_ids = []
    for plines in pages:
        stream_lines = ["BT", "/F1 9 Tf"]
        for (py, txt) in plines:
            safe = esc(txt)
            stream_lines.append(f"1 0 0 1 {MARGIN} {py} Tm ({safe}) Tj")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines)

        sid = add(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        pid = add(
            f"<< /Type /Page /Parent 2 0 R "
            f"/MediaBox [0 0 612 {PAGE_H}] "
            f"/Contents {sid} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>"
        )
        page_ids.append(pid)

    # Fix pages object (obj 2)
    kids    = " ".join(f"{p} 0 R" for p in page_ids)
    objs[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode()

    with open(path, "wb") as f:
        offsets = []
        pos = 0

        def write(data: bytes):
            nonlocal pos
            f.write(data)
            pos += len(data)

        write(b"%PDF-1.4\n")
        for i, obj in enumerate(objs, 1):
            offsets.append(pos)
            write(f"{i} 0 obj\n".encode())
            write(obj)
            write(b"\nendobj\n")

        xref_pos = pos
        write(f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode())
        for off in offsets:
            write(f"{off:010d} 00000 n \n".encode())
        write(f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode())


if __name__ == "__main__":
    generate()
