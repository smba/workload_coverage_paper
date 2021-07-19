public void insertRows(String[] rows) {
%\cova%	if (Configuration.DUPLICATE_CHECK) {
%\cova%		rows = new HashSet<String>(
%\cova%			Arrays.asList(array)
%\cova%		).toArray(new String[0]);
%\cova%	}
%\covc%	if (rows.length() > 50) {
%\covc%		this.insertBulkRows(rows);
%\covc%	} else {
%\covc%		for (String row: rows) {
%\covc%			this.insertRow(row);
%\covb%			if (Configuration.AUTOCOMMIT) {
%\covb%				this.commit();
%\covb%			}	
%\covc%		}
%\covc%	}
}