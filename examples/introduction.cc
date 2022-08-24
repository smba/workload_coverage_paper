public void insertRows(String[] rows) {
%\cova%	if (Configuration.DUPLICATE_CHECK) {
%\cova%		rows = new HashSet<String>(
%\cova%			Arrays.asList(array)
%\cova%		).toArray(new String[0]);
%\cova%	}
	if (rows.length() > 50) {
		this.insertBulkRows(rows);
	} else {
		for (String row: rows) {
			this.insertRow(row);
%\covb%			if (Configuration.AUTOCOMMIT) {
%\covb%				this.commit();
%\covb%			}	
		}
	}
}