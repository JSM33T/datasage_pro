import { Routes } from '@angular/router';
import { DocumentManagementComponent } from './components/document-management-component/document-management-component';
import { HomeComponent } from './components/home-component/home-component';
import { ChatComponent } from './components/chat-component/chat-component';

export const routes: Routes = [
	{
		path: '',
		component: HomeComponent
	},
	{
		path: 'documents',
		component: DocumentManagementComponent
	},
	{
		path: 'chat',
		component: ChatComponent
	}
];
