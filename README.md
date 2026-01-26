# How to run API

Run the following command:

```bash
$ uvicorn main:app --reload
```

# Requeriments

You will need some dependencies. To solve that, run:

```bash
$ pip install -r requirements.txt
```

# Environmental variables

The following variables are strictly required for connection to Supabase. You must create a .env file with:

```bash
SUPABASE_URL=<YOUR_SUPABASE_URL>
SUPABASE_APIKEY_SERVICE_ROLE=<YOUR_SUPABASE_APIKEY_SERVICE_ROLE>
```


