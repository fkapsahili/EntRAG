import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import click
import requests


@click.command()
@click.option("--days-back", default=30, help="Number of days to look back")
@click.option("--output", default="datasets/json/sec_filings_data.json", help="Output JSON file")
@click.option("--filing-types", default="10-K,10-Q,8-K,DEF 14A", help="Comma-separated filing types")
@click.option("--include-content", is_flag=True, help="Download actual filing content")
@click.option(
    "--companies",
    default="0000320193:Apple Inc,0000789019:Microsoft Corp,0001652044:Alphabet Inc",
    help="CIK:Company Name pairs, comma-separated",
)
@click.option("--email", required=True, help="Your email for SEC User-Agent")
def prefetch_sec_data(days_back, output, filing_types, include_content, companies, email):
    """Prefetch SEC EDGAR data for mock API"""

    headers = {"User-Agent": f"Mock API Prefetch {email}", "Accept-Encoding": "gzip, deflate", "Host": "data.sec.gov"}

    test_companies = {}
    for company_pair in companies.split(","):
        cik, name = company_pair.split(":")
        test_companies[cik] = name

    filing_type_list = [ft.strip() for ft in filing_types.split(",")]
    mock_data = {}

    with click.progressbar(filing_type_list, label="Fetching filings") as bar:
        for filing_type in bar:
            filings = fetch_recent_filings(filing_type, days_back, test_companies, headers, include_content)
            mock_data[filing_type] = filings

    output_path = Path(output)
    with output_path.open("w") as f:
        json.dump(mock_data, f, indent=2)

    total_filings = sum(len(v) for v in mock_data.values())
    click.echo(f"✓ Prefetched {total_filings} filings to {output_path}")


def fetch_recent_filings(filing_type, days_back, companies, headers, include_content):
    filings = []

    for cik, company_name in companies.items():
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            time.sleep(0.1)

            data = response.json()
            recent_filings = data["filings"]["recent"]

            for i in range(len(recent_filings["form"])):
                if recent_filings["form"][i] == filing_type:
                    filing_date = recent_filings["filingDate"][i]
                    if is_recent(filing_date, days_back):
                        filing = {
                            "company": company_name,
                            "cik": cik,
                            "filing_type": filing_type,
                            "filing_date": filing_date,
                            "accession_number": recent_filings["accessionNumber"][i],
                            "primary_document": recent_filings["primaryDocument"][i],
                            "document_url": f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{recent_filings['accessionNumber'][i].replace('-', '')}/{recent_filings['primaryDocument'][i]}",
                        }

                        if include_content:
                            filing = fetch_filing_content(filing, headers)

                        filings.append(filing)

        except requests.exceptions.RequestException as e:
            click.echo(f"Error fetching {company_name}: {e}", err=True)

    return filings


def is_recent(filing_date, days_back):
    filing_dt = datetime.strptime(filing_date, "%Y-%m-%d")
    cutoff_dt = datetime.now() - timedelta(days=days_back)
    return filing_dt >= cutoff_dt


def fetch_filing_content(filing, headers):
    try:
        response = requests.get(filing["document_url"], headers=headers)
        response.raise_for_status()
        time.sleep(0.1)
        filing["content_preview"] = response.text[:1000]
    except requests.exceptions.RequestException:
        filing["content_preview"] = None

    return filing


if __name__ == "__main__":
    prefetch_sec_data()
