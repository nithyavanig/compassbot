/*1614766352000*/












AUI.add(
	'portal-available-languages',
	function(A) {
		var available = {};

		var direction = {};

		

			available['en_US'] = 'English (United States)';
			direction['en_US'] = 'ltr';

		

			available['hi_IN'] = 'Hindi (India)';
			direction['hi_IN'] = 'ltr';

		

		Liferay.Language.available = available;
		Liferay.Language.direction = direction;
	},
	'',
	{
		requires: []
	}
);